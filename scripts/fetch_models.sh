#!/bin/sh
# Download the model weights the golden tests use into testdata/models, at
# pinned Hugging Face revisions, and check them against known hashes. Tests
# that need a model skip when it is missing.
#
#   scripts/fetch_models.sh            # all models (~1 GB)
#   scripts/fetch_models.sh minilm     # just all-MiniLM-L6-v2 (~90 MB)
#   scripts/fetch_models.sh t5         # just t5-small (~240 MB)
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
