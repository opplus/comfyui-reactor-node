import sys
import os
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
import threading
import traceback
import time

model_init_lock = threading.Lock()
D_FACE_RESTORE_MODEL=None
class FaceRestoreWorker:
    def __init__(self):
        self.face_helper = None
        self.FACE_SIZE = 512
        self.stop_event=False


    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            thread_num=4
    ):
        result = input_image
        t0=time.time()
        threads = []
        try:
            if face_restore_model != "none" and not model_management.processing_interrupted():
                faceSize = self._get_face_size(face_restore_model)

                logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

                model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
                device = model_management.get_torch_device()
                logger.status(f"mode get_device:{device}")
                device_id=os.getenv("CUDA_VISIBLE_DEVICES")
                # device=f"cuda:{device_id}"
                logger.status(f"restore_face actural device {device_id}")
                image_np = 255. * input_image.numpy()
                total_images = image_np.shape[0]
                out_images = []

                # 初始化模型
                global D_FACE_RESTORE_MODEL
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
                    D_FACE_RESTORE_MODEL=facerestore_model
                elif ".onnx" in face_restore_model:
                    ort_session = set_ort_session(model_path, providers=providers)
                    D_FACE_RESTORE_MODEL = ort_session
                else:
                    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                    facerestore_model = model_loading.load_state_dict(sd).eval()
                    facerestore_model.to(device)
                    D_FACE_RESTORE_MODEL=facerestore_model

                # 创建任务队列和结果队列
                task_queue = Queue()
                result_queue = Queue()

                            # 将图像放入队列
                for i in range(total_images):
                    task_queue.put((i, image_np[i, :, :, ::-1]))

                ptag=1
                # 启动线程
                # for _ in range(thread_num):
                #     t = threading.Thread(target=self.worker_process, args=(
                #         f"sub{ptag}",
                #         task_queue,
                #         result_queue,
                #         model_path,
                #         face_restore_model,
                #         face_restore_visibility,
                #         codeformer_weight,
                #         facedetection,
                #         faceSize,
                #         device,0
                #     ))
                #     t.start()
                #     threads.append(t)
                #     ptag=ptag+1


                # 主线程也参与处理
                self.worker_process(
                        "main",
                        task_queue,
                        result_queue,
                        model_path,
                        face_restore_model,
                        face_restore_visibility,
                        codeformer_weight,
                        facedetection,
                        faceSize,
                        device,
                        thread_num=thread_num,
                        timeout=180
                )

                # # 主线程等待所有任务完成
                # while not task_queue.qsize()<=0:
                #     pass

                # # 等待所有线程结束
                # for t in threads:
                #     t.join()

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
        except Exception as e:
            # # 等待所有线程结束
            self.stop_event=True
            raise e
        finally:
            D_FACE_RESTORE_MODEL=None
            logger.status(f"restore_face success cost:{time.time()-t0}")
        return result

    def worker_process(
            self,
            ptag,
            task_queue,
            result_queue,
            model_path,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device,
            thread_num=0,
            timeout=0
    ): 
        global D_FACE_RESTORE_MODEL
        device_id=os.getenv("CUDA_VISIBLE_DEVICES")
        logger.status(f"{ptag} worker_process enter qsize:{task_queue.qsize()}, device_id {device_id}, model_path:{model_path}, face_restore_model:{face_restore_model}, device:{device}")
        facerestore_model = D_FACE_RESTORE_MODEL
        face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection,save_ext='png', use_parse=True, device=device)
        s_cnt=0
        f_cnt=0
        t0=time.time()
        submit_t=0
        while True:
            t1=time.time()
            if self.stop_event:
                logger.status(f"{ptag} stop_event success:{s_cnt} fail:{f_cnt} cost:{t1-t0}")
                del face_helper
                return 
            ql=task_queue.qsize()
            if ql<=0:
                logger.status(f"{ptag} finish success:{s_cnt} fail:{f_cnt} cost:{t1-t0}")
                del face_helper
                return
            index, cur_image_np = task_queue.get()
            cost=time.time()-t0
            if timeout>0 and cost-timeout>0:
                raise ValueError(f"timeout:{timeout} cost:{cost}")
            try:
                if index is not None or cur_image_np is None:
                    logger.status(f"{ptag} restore {index} remine:{ql}")
                    result = self._process_single_image(
                        face_helper,
                        cur_image_np,
                        face_restore_model,
                        facerestore_model,
                        face_restore_visibility,
                        codeformer_weight,
                        facedetection,
                        faceSize,
                        device
                    )
                    s_cnt=s_cnt+1
                    result_queue.put((index, result))
                    if thread_num is not None and thread_num>=1 and submit_t<thread_num:
                        t = threading.Thread(target=self.worker_process, args=(
                            f"sub{submit_t}",
                            task_queue,
                            result_queue,
                            model_path,
                            face_restore_model,
                            face_restore_visibility,
                            codeformer_weight,
                            facedetection,
                            faceSize,
                            device,0,0
                        ))
                        t.start()       
                        submit_t=submit_t+1

                else:
                    s_cnt=s_cnt+1
                    result_queue.put((index, cur_image_np))
                    logger.status(f"{ptag} index or cur_image_np is null {index} {cur_image_np}")
            except Exception as error:
                logger.status(f"_process_single_image error {ptag}, {index}, {error}")
                err_msg=traceback.format_exc()
                logger.error(f"_process_single_image ",exc_info=True)
                result_queue.put((index, cur_image_np))
                f_cnt=f_cnt+1


    def _get_face_size(self, face_restore_model):
        if "1024" in face_restore_model.lower():
            return 1024
        elif "2048" in face_restore_model.lower():
            return 2048
        else:
            return 512

    def _pre_warmup(
            self,task_queue,result_queue,
            face_helper,
            face_restore_model,
            facerestore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device
    ):
        index, cur_image_np = task_queue.get()
        result = self._process_single_image(
            face_helper,
            cur_image_np,
            face_restore_model,
            facerestore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device
        )
        result_queue.put((index, result))

    def _process_single_image(
            self,
            face_helper,
            cur_image_np,
            face_restore_model,
            facerestore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
            faceSize,
            device
    ):
        original_resolution = cur_image_np.shape[0:2]

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
                    if ".onnx" in face_restore_model:
                        ort_session_inputs={}
                        ort_session=facerestore_model
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
