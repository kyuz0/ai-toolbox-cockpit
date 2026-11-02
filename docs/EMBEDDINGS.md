# EmbeddingGemma 2 with llama.cpp

In **Models → llama.cpp**, select **EmbeddingGemma 2 — text / multimodal embeddings**. On platforms with a tested-model filter, choose **All curated repositories** first. Download `embeddinggemma-2-Q8_0.gguf` from `ggml-org/embeddinggemma-2-GGUF`. For images and audio, also download `mmproj-embeddinggemma-2-Q8_0.gguf` from the same file list. The catalogue pins both selections to the same repository revision.

In **Server Mode → llama.cpp**, choose a current mainline toolbox and the downloaded model. Cockpit automatically selects the **Embeddings** profile: mean pooling, an 8,192-token context, batch and ubatch sizes of 8,192, and one parallel sequence. The command includes `--embedding --pooling mean`. Inputs must fit in one physical batch; Cockpit checks the context/batch allocation before launch.

Leave **Multimodal projector** empty for text only, or select the matching Q8 projector for images and audio. Start the server and send requests to `/v1/embeddings`. Each input returns a normalized 768-dimensional vector. Sampling and thinking settings do not apply.

Text search example, using the default localhost port:

```sh
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["task: search result | query: What is the capital of France?","title: Paris | text: Paris is the capital of France."],"encoding_format":"float"}'
```

Use the model's task prefixes in client input: `task: search result | query: …` for search queries and `title: none | text: …` for documents without a title. The server does not insert these prefixes automatically.

With the projector loaded, llama.cpp also accepts objects in the input list:

```json
{
  "input": [
    {"content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "A product photograph"}
    ]},
    {"content": [
      {"type": "input_audio", "input_audio": {"data": "...base64 WAV...", "format": "wav"}}
    ]}
  ],
  "encoding_format": "float"
}
```

Multimodal input objects are a llama.cpp extension to the OpenAI embeddings format. Text, images, audio and combined text/image requests were smoke-tested with the Q8 model/projector on Strix Halo ROCm 10.0, llama.cpp build 11870 (`74bca3950`). Audio remains experimental in llama.cpp; the smoke establishes encoder execution and valid output, not audio retrieval quality. Video and other hardware backends have not been qualified for this profile.

Sources: [official GGUFs](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), [model guidance](https://huggingface.co/google/embeddinggemma-2), [llama.cpp embeddings API](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#post-v1embeddings-openai-compatible-embeddings-api).
