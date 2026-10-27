#!/bin/sh
# Download the model weights the golden tests use into testdata/models, at
# pinned Hugging Face revisions, and check them against known hashes. Tests
# that need a model skip when it is missing.
#
#   scripts/fetch_models.sh            # all models (~3.2 GB)
#   scripts/fetch_models.sh minilm     # just all-MiniLM-L6-v2 (~90 MB)
#   scripts/fetch_models.sh t5         # just t5-small (~240 MB)
#   scripts/fetch_models.sh minicpm5   # just MiniCPM5-1B (~2.2 GB, bf16)
#   scripts/fetch_models.sh minicpm5-2b # MiniCPM5-2B (~5 GB, bf16; not in "all")
#   scripts/fetch_models.sh qwen3      # Qwen3-1.7B (~4.1 GB, bf16; not in "all")
set -eu
cd "$(dirname "$0")/.."

fetch() { # name repo revision "sha256  file" lines...
	name=$1 repo=$2 rev=$3
	shift 3
	dir=testdata/models/$name
	mkdir -p "$dir"
	for line in "$@"; do
		file=${line##* }
		if [ ! -s "$dir/$file" ]; then
			echo "fetching $repo $file"
			curl -fsSL --retry 5 -o "$dir/$file" "https://huggingface.co/$repo/resolve/$rev/$file"
		fi
	done
	(cd "$dir" && printf '%s\n' "$@" | shasum -a 256 -c -)
}

want=${1:-all}

if [ "$want" = all ] || [ "$want" = minilm ]; then
	fetch all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
		'953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41  config.json' \
		'53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db  model.safetensors' \
		'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  vocab.txt' \
		'acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b  tokenizer_config.json'
fi

if [ "$want" = all ] || [ "$want" = mxbai ]; then
	fetch mxbai-embed-large-v1 mixedbread-ai/mxbai-embed-large-v1 b33106f585b9ce46904ad7443a3b52b7a63e231c \
		'ce13f118fc183c005236901d2662ced752de632ebb161e6af800aadda2d176f7  config.json' \
		'36bfa45da00eb762ef9feebe4cff315ec779efadd08da11846bef5ba5b59b8f8  model.safetensors' \
		'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  vocab.txt' \
		'0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53  tokenizer_config.json'
fi

if [ "$want" = all ] || [ "$want" = t5 ]; then
	fetch t5-small google-t5/t5-small df1b051c49625cf57a3d0d8d3863ed4d13564fe4 \
		'530e25060e3a8d5f7b0fcf53bfea9f3601165161f3b1914676e98d97cf07bcf1  config.json' \
		'bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc  model.safetensors' \
		'd2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066  tokenizer.json' \
		'd1e7146101aa96282057f374aa9c3b260fba2109e5990edb1775d6efc70ffa3c  tokenizer_config.json'
fi

if [ "$want" = all ] || [ "$want" = minicpm5 ]; then
	fetch minicpm5-1b openbmb/MiniCPM5-1B 87179e5c1f455ef22e6223592d2d61351b525bfc \
		'6a6509b646cb3169616c5ffc3196e7ccaf9d4d6bc17b266581d241a31c217714  config.json' \
		'92afd6424501426eddcf7e1542f013d19e5987544977b4ee7bd26359bd5fd2ab  generation_config.json' \
		'7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d  model-00000-of-00001.safetensors' \
		'162add042e75abc3d571c4a8679523fa4f1ffc55d1fea25fc6658a19d6e957ee  model.safetensors.index.json' \
		'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81  tokenizer.json' \
		'094efb3cf1ff412284cc5945fc99dff58673a912760d04483a04aa1c716f66fd  tokenizer_config.json' \
		'82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db  special_tokens_map.json' \
		'7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c  chat_template.jinja'
fi

if [ "$want" = minicpm5-2b ]; then
	fetch minicpm5-2b openbmb/MiniCPM5-2B f97400052a43d642bbc6e9975e2397e3ae6a6b52 \
		'59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180  config.json' \
		'9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1  generation_config.json' \
		'14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c  model-00000-of-00001.safetensors' \
		'6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756  model.safetensors.index.json' \
		'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81  tokenizer.json' \
		'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b  tokenizer_config.json' \
		'82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db  special_tokens_map.json' \
		'bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd  chat_template.jinja'
fi

if [ "$want" = qwen3 ]; then
	fetch qwen3-1.7b Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
		'1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197  config.json' \
		'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json' \
		'0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d  model.safetensors.index.json' \
		'169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5  model-00001-of-00002.safetensors' \
		'912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9  model-00002-of-00002.safetensors' \
		'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json' \
		'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  tokenizer_config.json' \
		'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json' \
		'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt' \
		'832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  LICENSE'
fi
