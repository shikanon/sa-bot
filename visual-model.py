import time
from volcengine.visual.VisualService import VisualService
import json
from typing import Dict, Union, Optional

class VisualServiceHandler:
    def __init__(self, ak: str, sk: str):
        """
        初始化视觉服务处理类
        
        参数:
            ak (str): 您的Access Key (访问密钥)
            sk (str): Your Secret Key (密钥)
        """
        self.visual_service = VisualService()
        self.visual_service.set_socket_timeout(3600)
        self.visual_service.set_connection_timeout(3600)
        self.visual_service.set_ak(ak)
        self.visual_service.set_sk(sk)
        self.default_action = "CVSubmitTask"
        self.default_version = "2022-08-31"
        self.visual_service.set_api_info(self.default_action, self.default_version)

    # 形象创建相关方法
    def create_avatar(self, image_url: str, req_key: str = "realman_avatar_picture_create_role_loopy") -> Dict:
        """
        提交虚拟形象创建任务（完整版）
        
        参数:
            image_url (str): 图片URL链接
            req_key (str): 服务标识，默认为loopy模式
            
        返回:
            dict: 包含完整响应信息的字典
        """
        return self._process_creation_request(image_url, req_key, verbose=True)

    def submit_creation_task(self, image_url: str, req_key: str = "realman_avatar_picture_create_role_loopy") -> Union[str, Dict]:
        """
        简化版提交形象创建任务
        
        参数:
            image_url (str): 图片URL链接
            req_key (str): 服务标识，默认为loopy模式
            
        返回:
            str: 成功时返回task_id，失败时返回错误信息字典
        """
        result = self._process_creation_request(image_url, req_key, verbose=False)
        return result['task_id'] if result['success'] else result['error']

    def _process_creation_request(self, image_url: str, req_key: str, verbose: bool) -> Dict:
        """内部处理形象创建请求"""
        form = {"req_key": req_key, "image_url": image_url}
        resp = self.visual_service.cv_json_api(self.default_action, form)
        
        return self._format_response(resp, verbose, "形象创建")

    # 视频生成相关方法
    def generate_video(self, resource_id: str, audio_url: str, req_key: str = "realman_avatar_picture_loopy") -> Dict:
        """
        提交视频生成任务（完整版）
        
        参数:
            resource_id (str): 形象ID（从创建形象接口获取）
            audio_url (str): 音频URL链接
            req_key (str): 服务标识，默认为loopy模式
            
        返回:
            dict: 包含完整响应信息的字典
        """
        return self._process_generation_request(resource_id, audio_url, req_key, verbose=True)

    def submit_generation_task(self, resource_id: str, audio_url: str, req_key: str = "realman_avatar_picture_loopy") -> Union[str, Dict]:
        """
        简化版提交视频生成任务
        
        参数:
            resource_id (str): 形象ID（从创建形象接口获取）
            audio_url (str): 音频URL链接
            req_key (str): 服务标识，默认为loopy模式
            
        返回:
            str: 成功时返回task_id，失败时返回错误信息字典
        """
        result = self._process_generation_request(resource_id, audio_url, req_key, verbose=False)
        return result['task_id'] if result['success'] else result['error']

    def _process_generation_request(self, resource_id: str, audio_url: str, req_key: str, verbose: bool) -> Dict:
        """内部处理视频生成请求"""
        form = {
            "req_key": req_key,
            "resource_id": resource_id,
            "audio_url": audio_url
        }
        resp = self.visual_service.cv_json_api(self.default_action, form)
        
        return self._format_response(resp, verbose, "视频生成")

    # 查询任务相关方法
    def query_creation_result(self, task_id: str, req_key: str = "realman_avatar_picture_create_role_loopy") -> Dict:
        """
        查询形象创建任务结果
        
        参数:
            task_id (str): 任务ID
            req_key (str): 服务标识，需与提交时一致
            
        返回:
            dict: 包含任务结果的字典
        """
        return self._query_task_result(task_id, req_key, "形象创建")

    def query_generation_result(self, task_id: str, req_key: str = "realman_avatar_picture_loopy", 
                              max_retries: int = 30, interval: float = 2.0) -> Dict:
        """
        查询视频生成任务结果（带轮询等待）
        
        参数:
            task_id (str): 任务ID
            req_key (str): 服务标识，需与提交时一致
            max_retries (int): 最大重试次数，默认30次
            interval (float): 每次查询间隔时间(秒)，默认2秒
            
        返回:
            dict: 包含任务结果的字典
        """
        self.visual_service.set_api_info(self.default_action, self.default_version)
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                result = self._query_task_result(task_id, req_key, "视频生成")
                
                # 如果查询成功且任务已完成
                if result.get('success') and result.get('status') == 'done':
                    return result
                
                # 如果查询成功但任务未完成
                if result.get('success'):
                    print(result)
                    print(f"任务处理中，进度: {result.get('progress', 0)}%，等待中... (尝试 {retries + 1}/{max_retries})")
                else:
                    # 其他错误直接返回
                    print(f"查询出错: {result.get('error', '未知错误')}")
                    return result
            
                time.sleep(interval)
                retries += 1
                
            except Exception as e:
                last_error = str(e)
                print(f"查询出错: {last_error}, 等待中... (尝试 {retries + 1}/{max_retries})")
                time.sleep(interval)
                retries += 1
        
        # 达到最大重试次数仍未成功
        return {
            "success": False,
            "error": f"达到最大重试次数({max_retries})仍未获取结果。最后错误: {last_error or '未知错误'}",
            "task_id": task_id
        }

    def _query_task_result(self, task_id: str, req_key: str, task_type: str) -> Dict:
        """内部处理查询任务请求"""
        form = {"req_key": req_key, "task_id": task_id}
        
        # 临时修改API动作为查询
        self.visual_service.set_api_info("CVGetResult", self.default_version)
        try:
            resp = self.visual_service.cv_json_api("CVGetResult", form)
        except Exception as e:
            # 捕获异常并转换为错误响应
            error_msg = str(e)
            if "50215" in error_msg:
                return {
                    "success": False,
                    "error": "50215: Input invalid for this service (可能任务还未准备好)",
                    "task_id": task_id
                }
            return {
                "success": False,
                "error": error_msg,
                "task_id": task_id
            }
        finally:
            # 恢复默认API动作
            self.visual_service.set_api_info(self.default_action, self.default_version)
        
        result = self._format_response(resp, False, f"{task_type}查询")
        
        # 解析resp_data中的资源ID或视频URL
        if result['success']:
            try:
                resp_data = resp.get("data", {})
                result['status'] = resp_data.get("status", "")
                
                # 尝试解析resp_data字符串
                if isinstance(resp_data.get("resp_data"), str):
                    resp_data_json = json.loads(resp_data.get("resp_data", "{}"))
                    result['progress'] = resp_data_json.get("progress", 0)
                    
                    if task_type == "形象创建":
                        result['resource_id'] = resp_data_json.get("resource_id", "")
                    else:
                        # 处理不同模式的视频URL返回方式
                        if "loopyb" in req_key:  # 大画幅模式
                            result['video_url'] = resp_data.get("video_url", "")
                        else:  # 普通和loopy模式
                            preview_urls = resp_data_json.get("preview_url", [])
                            result['video_url'] = preview_urls[0] if preview_urls else ""
                else:
                    result['error'] = "响应数据格式不正确"
                    result['success'] = False
                    
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                result['error'] = f"解析响应数据失败: {str(e)}"
                result['success'] = False
        
        return result

    # 通用方法
    def _format_response(self, resp: Dict, verbose: bool, action_name: str) -> Dict:
        """格式化响应结果"""
        success = resp.get("code") == 10000
        result = {
            "success": success,
            "task_id": resp.get("data", {}).get("task_id", ""),
            "request_id": resp.get("request_id", ""),
            "message": resp.get("message", ""),
            "error": "" if success else resp.get("message", "Unknown error"),
            "raw_response": resp
        }

        if verbose:
            self._print_result(result, action_name)
        return result

    def _print_result(self, result: Dict, action_name: str):
        """打印格式化结果"""
        print(f"\n{action_name}任务结果:")
        print(f"状态: {'成功' if result['success'] else '失败'}")
        if result['success']:
            print(f"任务ID: {result['task_id']}")
            print(f"请求ID: {result['request_id']}")
            if 'resource_id' in result:
                print(f"形象ID: {result['resource_id']}")
            if 'video_url' in result:
                print(f"视频URL: {result['video_url']}")
            if 'progress' in result:
                print(f"处理进度: {result['progress']}%")
        else:
            print(f"错误原因: {result['error']}")
        print(f"消息: {result['message']}")
