# vLLM Docker

ローカルに保存したモデルを Docker 上で扱い、必要に応じて量子化したうえで vLLM の OpenAI 互換 API として起動するための構成です。

## 構成

- `docker-compose.yml`
  - `quantize`: ローカルモデルを読み込み、量子化済みモデルを `./models` に保存します。
  - `vllm`: ローカルモデルまたは量子化済みモデルを読み込み、OpenAI 互換 API を `localhost:8000` で提供します。
- `quantize/Dockerfile.qwen3_5_quantize`
  - 現在の量子化ジョブ用イメージです。Qwen3.5 系で必要な Transformers / LLM Compressor 依存を含みます。
- `quantize/quantize_qwen3_5.py`
  - LLM Compressor の `oneshot` を使って GPTQ W4A16 量子化を実行します。

## 前提

- Docker Desktop
- NVIDIA GPU
- NVIDIA Container Toolkit
- 使用するモデルをローカルの `./models` 配下に配置済みであること

現在の設定例では、次のディレクトリ構成を想定しています。

```text
models/
  Qwen3.5-2B/
    config.json
    model.safetensors...
    tokenizer.json
    preprocessor_config.json
    video_preprocessor_config.json
```

## モデル設定

設定値は環境変数ではなく、`docker-compose.yml` に直接書いています。

現在の入力モデルと量子化出力先は以下です。

```yaml
MODEL_ID: /models/Qwen3.5-2B
QUANTIZED_MODEL_DIR: /models/Qwen3.5-2B-GPTQ-W4A16-G128
```

vLLM 側は量子化済みモデルを読む設定です。

```yaml
- "--model=/models/Qwen3.5-2B-GPTQ-W4A16-G128"
- "--served-model-name=Qwen3.5-2B-GPTQ-W4A16-G128"
```

別のモデルを使う場合は、`docker-compose.yml` の `quantize.environment` と `vllm.command` を同じモデル名に合わせて編集してください。

例:

```yaml
MODEL_ID: /models/YourModel
QUANTIZED_MODEL_DIR: /models/YourModel-GPTQ-W4A16-G128
```

```yaml
- "--model=/models/YourModel-GPTQ-W4A16-G128"
- "--served-model-name=YourModel-GPTQ-W4A16-G128"
```

量子化せずに vLLM で直接起動したい場合は、`vllm.command` の `--model` を元モデルのディレクトリに向けます。

```yaml
- "--model=/models/YourModel"
- "--served-model-name=YourModel"
```

## 量子化イメージをビルド

初回、または `quantize/` 配下を変更した後はビルドします。

```powershell
docker compose --profile quantize build quantize
```

依存関係を確実に入れ直したい場合はキャッシュなしでビルドします。

```powershell
docker compose --profile quantize build --no-cache quantize
```

## 量子化を実行

```powershell
docker compose --profile quantize run --rm quantize
```

成功すると、現在の設定例では以下に量子化済みモデルが保存されます。

```text
models/
  Qwen3.5-2B-GPTQ-W4A16-G128/
```

量子化時は `HuggingFaceH4/ultrachat_200k` をキャリブレーションデータとして使います。未認証リクエストの警告が出る場合がありますが、公開データセットを取得できていればそのまま進みます。

## 画像入力について

現在の Qwen3.5 系の設定例では、vision / video 関連レイヤを量子化対象から除外しています。

```yaml
GPTQ_IGNORE: "lm_head,re:.*visual.*,re:.*linear_attn.*"
```

量子化後のディレクトリには、画像・動画入力に必要な以下の設定ファイルを元モデルからコピーします。

```text
chat_template.jinja
preprocessor_config.json
processor_config.json
image_processor_config.json
video_preprocessor_config.json
generation_config.json
```

別の VLM を使う場合は、そのモデルで必要な processor 設定ファイルや量子化対象外レイヤを確認してください。

## vLLM API サーバを起動

量子化が完了したら、vLLM を起動します。

```powershell
docker compose --profile serve up vllm
```

バックグラウンドで起動したい場合:

```powershell
docker compose --profile serve up -d vllm
```

停止する場合:

```powershell
docker compose --profile serve down
```

## 動作確認

モデル一覧を確認します。

```powershell
curl http://localhost:8000/v1/models `
  -H "Authorization: Bearer sk-local-CHANGE_ME"
```

テキストチャットの例:

```powershell
curl http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer sk-local-CHANGE_ME" `
  -d '{
    "model": "Qwen3.5-2B-GPTQ-W4A16-G128",
    "messages": [
      {"role": "user", "content": "こんにちは。短く自己紹介してください。"}
    ],
    "max_tokens": 128
  }'
```

`model` には `--served-model-name` で指定した名前を入れてください。

## よく使う設定

`docker-compose.yml` の以下を必要に応じて編集します。

```yaml
NUM_CALIBRATION_SAMPLES: "512"
CALIBRATION_MAX_SEQUENCE_LENGTH: "2048"
```

vLLM のメモリ使用量や同時処理数:

```yaml
- "--gpu-memory-utilization=0.7"
- "--max-num-seqs=4"
```

API キー:

```yaml
- "--api-key=sk-local-CHANGE_ME"
```

## 注意点

- `MODEL_ID` と `QUANTIZED_MODEL_DIR` は同じパスにしないでください。元モデルを上書きしないためです。
- `quantize/quantize_qwen3_5.py` を変更した場合は、量子化イメージの再ビルドが必要です。
- vLLM で量子化済みモデルを使う場合は、`--model` に量子化済みディレクトリを指定します。
- 量子化せずに serve する場合は、`--model` に元モデルのディレクトリを指定します。
- `models/` は `.gitignore` 対象です。モデル本体や量子化済み checkpoint は Git 管理しません。
