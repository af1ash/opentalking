# 通过 OmniRT 使用 FeatherTalk

模型、FeatherHuBERT 和人物素材在 OmniRT 服务端加载；OpenTalking 只负责对话、TTS、
音频分块、WebSocket 通信和 WebRTC 播放。

先按 OmniRT 的 `docs/user_guide/serving/feathertalk_ws.md` 配置并启动 FeatherTalk。
确认 `GET /v1/audio2video/models` 中 `feathertalk` 为已连接状态。

## OpenTalking 配置

在 `.env` 中设置：

```dotenv
OMNIRT_ENDPOINT=http://127.0.0.1:9000
OPENTALKING_FEATHERTALK_BACKEND=omnirt
OPENTALKING_DEFAULT_MODEL=feathertalk
```

重启服务，前端选择 FeatherTalk。也可以使用现有脚本：

```bash
bash scripts/start_unified.sh --backend omnirt --model feathertalk --omnirt http://127.0.0.1:9000
```

如需直接指定 WebSocket 地址：

```dotenv
OPENTALKING_FEATHERTALK_BACKEND=direct_ws
OPENTALKING_FEATHERTALK_WS_URL=ws://127.0.0.1:9000/v1/audio2video/feathertalk
```

## 人物素材与权重

默认使用 OmniRT 的 `OMNIRT_FEATHERTALK_DATASET`、`OMNIRT_FEATHERTALK_CHECKPOINT`、
`OMNIRT_FEATHERTALK_HUBERT_CHECKPOINT`。OpenTalking 中保留形象 manifest 和 reference.png/jpg 即可。

也可在形象 manifest 中设置：

```json
{
  "metadata": {
    "feathertalk": {
      "dataset": "/srv/feathertalk/data/person",
      "checkpoint": "/srv/feathertalk/checkpoints/person/last.pth",
      "hubert_checkpoint": "/srv/feathertalk/feather_hubert.pth"
    }
  }
}
```

路径指向 **OmniRT 服务端文件系统**，OpenTalking 不检查或转换它们。
请求携带的路径需在 OmniRT 的 `OMNIRT_ALLOWED_FRAME_ROOTS` 范围内。
也支持 `models.feathertalk` YAML 配置和 `OPENTALKING_FEATHERTALK_DATASET`、
`OPENTALKING_FEATHERTALK_CHECKPOINT`、`OPENTALKING_FEATHERTALK_HUBERT_CHECKPOINT` 环境变量。
显式环境/Settings 配置优先于形象 metadata，再使用模型 YAML 默认配置。

## 实时协议

- `init` / `init_ok` 初始化会话，使用 16kHz 单声道 int16，25fps。
- 每块音频为 640 采样点（40ms），以 `AUDI + PCM` 发送，接收 `VIDX + JPEG` 视频帧。
- 生成前预读约 400ms 的后续音频，通过 `audio_context` 传入真实右侧上下文。
  后续音频只参与特征提取，不重复播放。每轮对话重置音频历史。
- idle 使用远端 `idle` 请求动态播放人物素材。
- 音频、idle、初始化和关闭请求共用客户端锁，避免并发读取同一 WebSocket。
