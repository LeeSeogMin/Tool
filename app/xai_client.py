"""
xAI API Client

OpenAI 호환 인터페이스를 제공하는 X.AI API 클라이언트입니다.
이 모듈은 기존 코드가 OpenAI API를 사용하는 방식 그대로 X.AI API를 사용할 수 있도록 합니다.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union, Callable, AsyncIterator
import logging
import time
import asyncio

from app.logger import logger

class XAIClient:
    """X.AI API와 상호작용하기 위한 기본 클라이언트"""
    
    BASE_URL = "https://api.x.ai/v1"
    
    def __init__(self, api_key):
        """
        클라이언트 초기화
        
        Args:
            api_key (str): X.AI API 키
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def chat_completion(self, messages, model=None, temperature=0, stream=False, max_tokens=None):
        """
        채팅 완성 API 호출
        
        Args:
            messages (list): 'role'과 'content'를 가진 메시지 객체 리스트
            model (str): 사용할 모델 이름
            temperature (float): 샘플링 온도
            stream (bool): 스트리밍 응답 여부
            max_tokens (int, optional): 최대 토큰 수
            
        Returns:
            dict: API 응답
        """
        endpoint = f"{self.BASE_URL}/chat/completions"
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # 불필요한 None 값 제거
        payload = {k: v for k, v in payload.items() if v is not None}
        
        max_retries = 5
        retry_delay = 1  # 초기 대기 시간 (초)
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=self.headers, json=payload)
                
                # 429 오류 처리 (Too Many Requests)
                if response.status_code == 429:
                    wait_time = retry_delay * (2 ** attempt)  # 지수 백오프
                    logger.warning(f"API 속도 제한에 도달했습니다. {wait_time}초 대기 후 재시도합니다. (시도 {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()  # 다른 HTTP 오류에 대한 예외 발생
                return response.json()
                
            except requests.exceptions.RequestException as e:
                # 429 이외의 다른 오류이거나 마지막 시도인 경우
                if attempt == max_retries - 1:
                    logger.error(f"XAI API 호출 오류: {str(e)}")
                    raise
                
                # 네트워크 오류 등에도 재시도
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"API 호출 오류, {wait_time}초 후 재시도합니다: {str(e)}")
                time.sleep(wait_time)
        
        # 모든 재시도 실패 시
        raise Exception(f"XAI API 호출이 {max_retries}번의 시도 후에도 실패했습니다.")


# OpenAI 응답 객체와 호환되는 클래스들
class ChatCompletionMessage:
    """OpenAI의 응답 메시지와 호환되는 클래스"""
    
    def __init__(self, content=None, role=None, tool_calls=None, **kwargs):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls  # 실제 xAI는 도구 호출을 지원하지 않을 수 있지만, 인터페이스 호환을 위해 유지
        
        # 추가 필드
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatCompletionChoice:
    """OpenAI의 선택 객체와 호환되는 클래스"""
    
    def __init__(self, message=None, index=0, delta=None, **kwargs):
        if isinstance(message, dict):
            self.message = ChatCompletionMessage(**message)
        else:
            self.message = message
        self.index = index
        self.delta = delta
        
        # 추가 필드
        for key, value in kwargs.items():
            setattr(self, key, value)


class DeltaMessage:
    """스트리밍 응답의 델타 메시지"""
    
    def __init__(self, content=None, **kwargs):
        self.content = content
        
        # 추가 필드
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatCompletion:
    """OpenAI 응답과 호환되는 클래스"""
    
    def __init__(self, id=None, choices=None, model=None, object=None, **kwargs):
        """
        OpenAI 응답과 호환되는 객체 초기화
        
        Args:
            id (str): 응답 ID
            choices (list): 응답 선택지 목록
            model (str): 모델 이름
            object (str): 객체 타입
            **kwargs: 추가 필드 (created, usage 등)
        """
        self.id = id
        if choices and isinstance(choices, list):
            self.choices = [
                ChatCompletionChoice(**choice) if isinstance(choice, dict) else choice
                for choice in choices
            ]
        else:
            self.choices = []
        self.model = model
        self.object = object or "chat.completion"
        
        # 추가 필드 (예: created, usage 등)
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChunkResponse:
    """스트리밍 응답 청크"""
    
    def __init__(self, id=None, choices=None, model=None, **kwargs):
        self.id = id
        if choices and isinstance(choices, list):
            # choices에서 delta를 처리
            self.choices = []
            for choice in choices:
                if isinstance(choice, dict):
                    if 'delta' in choice:
                        delta = DeltaMessage(**choice['delta'])
                        self.choices.append(ChatCompletionChoice(delta=delta, index=choice.get('index', 0)))
                    else:
                        self.choices.append(ChatCompletionChoice(**choice))
                else:
                    self.choices.append(choice)
        else:
            self.choices = []
        self.model = model
        
        # 추가 필드
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatCompletionsAPI:
    """OpenAI의 chat.completions 구조와 호환되는 클래스"""
    
    def __init__(self, client):
        self.client = client
    
    async def create(self, model=None, messages=None, temperature=0, max_tokens=None, stream=False, **kwargs):
        """
        채팅 완성 생성 API 호출
        
        Args:
            model (str): 사용할 모델
            messages (list): 메시지 목록
            temperature (float): 샘플링 온도
            max_tokens (int, optional): 최대 토큰 수
            stream (bool): 스트리밍 응답 여부
            **kwargs: 무시될 추가 매개변수들
            
        Returns:
            ChatCompletion 또는 AsyncIterator[ChunkResponse]: 응답 객체 (스트리밍 시 이터레이터)
        """
        # OpenAI에서 지원하지만 xAI에서 지원하지 않는 매개변수들 로깅
        unsupported_params = [k for k in kwargs.keys() if k not in ['stream']]
        if unsupported_params:
            logger.warning(f"xAI API에서 지원하지 않는 매개변수가 무시됨: {', '.join(unsupported_params)}")
        
        # 스트리밍 응답 처리
        if stream:
            return self._create_streaming(model, messages, temperature, max_tokens)
        
        # 일반 응답 처리
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            # OpenAI 형식으로 응답 변환
            return ChatCompletion(**response)
            
        except Exception as e:
            logger.error(f"xAI API 호출 오류: {str(e)}")
            raise
    
    async def _create_streaming(self, model, messages, temperature, max_tokens):
        """
        스트리밍 응답을 처리하기 위한 비동기 제너레이터
        
        xAI가 실제로 스트리밍을 지원하지 않으므로 단일 응답을 스트리밍처럼 변환
        """
        try:
            # 스트리밍을 지원하지 않는 경우, 단일 응답을 가져와서 청크처럼 제공
            response = self.client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False  # 실제 스트리밍이 지원되지 않음
            )
            
            # 응답을 한 번에 하나의 청크로 제공
            content = response['choices'][0]['message']['content']
            
            # 콘텐츠를 단어 단위로 분할하여 스트리밍 시뮬레이션
            words = content.split(' ')
            
            for i, word in enumerate(words):
                chunk = {
                    'id': response.get('id', 'stream-id'),
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': word + (' ' if i < len(words) - 1 else '')},
                    }]
                }
                yield ChunkResponse(**chunk)
                
                # 실제 스트리밍처럼 약간의 지연 추가
                await asyncio.sleep(0.05)
            
        except Exception as e:
            logger.error(f"xAI 스트리밍 시뮬레이션 오류: {str(e)}")
            raise


class ChatAPI:
    """OpenAI의 chat 구조와 호환되는 클래스"""
    
    def __init__(self, client):
        self.client = client
        self.completions = ChatCompletionsAPI(client)


class AsyncXAI:
    """OpenAI와 호환되는 xAI 클라이언트"""
    
    def __init__(self, api_key=None, base_url=None):
        """
        클라이언트 초기화
        
        Args:
            api_key (str): X.AI API 키
            base_url (str): 사용되지 않음, OpenAI 호환성을 위해 유지
        """
        self.api_key = api_key
        self.client = XAIClient(api_key)
        self.chat = ChatAPI(self.client) 