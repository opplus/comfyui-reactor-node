import sys
from multiprocessing import Process, Queue, cpu_count

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from r_basicsr.utils.registry import ARCH_REGISTRY
from r_chainner import model_loading
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from reactor_utils import (
    img2tensor,
    tensor2img,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face
)
from scripts.reactor_faceswap import (
    providers
)
from scripts.reactor_logger import logger


class FaceRestoreWorker:
    def __init__(self):
        self.face_helper = None
        self.FACE_SIZE = 512

    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
    ):
        result = input_image

        if face_restore_model != "none" and not model_management.processing_interrupted():
            faceSize = self._get_face_size(face_restore_model)

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
            device = model_management.get_torch_device()

            image_np = 255. * input_image.numpy()
            total_images = image_np.shape[0]
            out_images = []

            # 创建任务队列和结果队列
            task_queue = Queue()
            result_queue = Queue()

            # 最多允许同时运行的子进程数量
            max_processes = min(cpu_count(), 10)

            # 启动子进程
            processes = []
            for _ in range(max_processes):
                p = Process(target=self.worker_process, args=(
                    task_queue,
                    result_queue,
                    model_path,
                    face_restore_model,
                    face_restore_visibility,
                    codeformer_weight,
                    facedetection,
                    faceSize,
                    device
                ))
                p.start()
                processes.append(p)

            # 将图像放入队列
            for i in range(total_images):
                task_queue.put((i, image_np[i, :, :, ::-1]))

            # 发送结束信号
            for _ in range(max_processes):
                task_queue.put(None)

            # 收集结果
            results = []
            processed_count = 0
            while processed_count < total_images:
                result_index, result_image = result_queue.get()
                results.append((result_index, result_image))
                processed_count += 1

            # 按照原始顺序排序结果
            results.sort(key=lambda x: x[0])
            results = [r[1] for r in results]

            restored_img_np = np.array(results).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

        return result

    def worker_process(
            self,
            task_queue,
            result_queue,
            model_path,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device
    ):
        facerestore_model = None
        ort_session = None
        ort_session_inputs = None
        if "codeformer" in face_restore_model.lower():
            codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            checkpoint = torch.load(model_path)["params_ema"]
            codeformer_net.load_state_dict(checkpoint)
            facerestore_model = codeformer_net.eval()
        elif ".onnx" in face_restore_model:
            ort_session = set_ort_session(model_path, providers=providers)
            ort_session_inputs = {}
            facerestore_model = ort_session
        else:
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            facerestore_model = model_loading.load_state_dict(sd).eval()
            facerestore_model.to(device)

        while True:
            index, cur_image_np = task_queue.get()
            if cur_image_np is None:
                break
            result = self._process_single_image(
                cur_image_np,
                face_restore_model,
                facerestore_model,
                ort_session,
                ort_session_inputs,
                face_restore_visibility,
                codeformer_weight,
                facedetection,
                faceSize,
                device
            )
            result_queue.put((index, result))

    def _get_face_size(self, face_restore_model):
        if "1024" in face_restore_model.lower():
            return 1024
        elif "2048" in face_restore_model.lower():
            return 2048
        else:
            return 512

    def _process_single_image(
            self,
            cur_image_np,
            face_restore_model,
            facerestore_model,
            ort_session,
            ort_session_inputs,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device
    ):
        original_resolution = cur_image_np.shape[0:2]

        face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection,
                                        save_ext='png', use_parse=True, device=device)

        face_helper.clean_all()
        face_helper.read_image(cur_image_np)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        restored_face = None

        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    if ".onnx" in facerestore_model:
                        for ort_session_input in ort_session.get_inputs():
                            if ort_session_input.name == "input":
                                cropped_face_prep = prepare_cropped_face(cropped_face)
                                ort_session_inputs[ort_session_input.name] = cropped_face_prep
                            if ort_session_input.name == "weight":
                                weight = np.array([1], dtype=np.double)
                                ort_session_inputs[ort_session_input.name] = weight

                        output = ort_session.run(None, ort_session_inputs)[0][0]
                        restored_face = normalize_cropped_face(output)

                    else:
                        output = facerestore_model(cropped_face_t, w=codeformer_weight)[
                            0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                del output
                torch.cuda.empty_cache()

            except Exception as error:
                print(f"\tFailed inference: {error}", file=sys.stderr)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            if face_restore_visibility < 1:
                restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)

        restored_img = face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        if original_resolution != restored_img.shape[0:2]:
            restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1] / restored_img.shape[1],
                                      fy=original_resolution[0] / restored_img.shape[0], interpolation=cv2.INTER_AREA)

        face_helper.clean_all()

        return restored_img


def _test():
    import torch

    # 创建输入图像
    input_image = torch.rand(100, 3, 512, 512)  # 假设有 100 张图像

    # 创建 FaceRestoreWorker 实例
    worker = FaceRestoreWorker()

    # 调用 restore_face 方法
    output_image = worker.restore_face(
        input_image,
        face_restore_model="codeformer",
        face_restore_visibility=0.5,
        codeformer_weight=0.5,
        facedetection="retinaface_resnet50"
    )
    print(output_image.shape)  # 应该输出 (100, 3, 512, 512)
