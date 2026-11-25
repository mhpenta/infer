#!/bin/sh
# Download the model weights the golden tests use into testdata/models, at
# pinned Hugging Face revisions, and check them against known hashes. Tests
# that need a model skip when it is missing.
# An interrupted download resumes from its .part file on the next run.
#
#   scripts/fetch_models.sh            # all models (~3.2 GB)
#   scripts/fetch_models.sh minilm     # just all-MiniLM-L6-v2 (~90 MB)
#   scripts/fetch_models.sh t5         # just t5-small (~240 MB)
#   scripts/fetch_models.sh minicpm5   # just MiniCPM5-1B (~2.2 GB, bf16)
#   scripts/fetch_models.sh minicpm5-2b # MiniCPM5-2B (~5 GB, bf16; not in "all")
#   scripts/fetch_models.sh qwen3      # Qwen3-1.7B (~4.1 GB, bf16; not in "all")
#   scripts/fetch_models.sh qwen3-4b   # Qwen3-4B (~8 GB, bf16; not in "all")
#   scripts/fetch_models.sh openjev-0.8b # OpenJev's Qwen3.5-0.8B NLI checkpoint (~1.7 GB, bf16; not in "all")
#   scripts/fetch_models.sh openjev-4b # OpenJev's Qwen3.5-4B NLI checkpoint, v5 (~9.1 GB, bf16; not in "all")
#   scripts/fetch_models.sh inkling2-mini-tokenizer # Inkling 2 Mini's tokenizer and chat template (~26 MB; not in "all")
#   scripts/fetch_models.sh inkling2-mini # Inkling 2 Mini (~19.5 GB, NVFP4/MXFP8 experts; not in "all")
#   scripts/fetch_models.sh yolos-tiny # YOLOS-Tiny object detector (~25 MB, f32; not in "all")
#   scripts/fetch_models.sh yolos-small-300 # YOLOS-Small-300 object detector (~117 MB, f32; not in "all")
set -eu
cd "$(dirname "$0")/.."

fetch() { # name repo[:subfolder] revision "sha256  file" lines...
	name=$1 repo=${2%%:*} rev=$3
	sub=${2#"$repo"}
	sub=${sub#:}
	sub=${sub:+$sub/}
	shift 3
	dir=testdata/models/$name
	mkdir -p "$dir"
	for line in "$@"; do
		file=${line##* }
		if [ ! -s "$dir/$file" ]; then
			echo "fetching $repo $sub$file"
			# Download into .part, each attempt resuming from its size: a
			# connection slower than 100 KB/s for a minute is dropped and
			# the next attempt opens a new one.
			tries=0
			until curl -fsSL -C - --speed-limit 100000 --speed-time 60 \
				-o "$dir/$file.part" "https://huggingface.co/$repo/resolve/$rev/$sub$file"; do
				tries=$((tries + 1))
				if [ "$tries" -ge 20 ]; then
					echo "giving up on $file after $tries attempts; rerun to resume" >&2
					exit 1
				fi
				echo "resuming $file ($(wc -c <"$dir/$file.part" | tr -d ' ') bytes so far)"
				sleep 5
			done
			mv "$dir/$file.part" "$dir/$file"
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

if [ "$want" = qwen3-4b ]; then
	fetch qwen3-4b Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c \
		'8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a  config.json' \
		'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json' \
		'6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2  model.safetensors.index.json' \
		'328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223  model-00001-of-00003.safetensors' \
		'6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5  model-00002-of-00003.safetensors' \
		'e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1  model-00003-of-00003.safetensors' \
		'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json' \
		'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  tokenizer_config.json' \
		'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json' \
		'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt' \
		'832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  LICENSE'
fi

if [ "$want" = openjev-0.8b ]; then
	fetch openjev-0.8b AlexWortega/openjev:qwen3.5-0.8b-nli-v2s-long a20448012c213128955ca0c693e7c943865cab77 \
		'3b914561e53dad38981a498af0b0bf430300be9d3e5fa3cd7b10c2049c073e52  config.json' \
		'cf6d62a341c0c804f9a926eec71aefc9859adb28978736e757b49bce35d9b8f8  model.safetensors' \
		'd73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20  tokenizer.json' \
		'289d88d849f89fcf0f9c145773122f6718095cd4aee05435afbd0293569c7085  tokenizer_config.json'
fi

if [ "$want" = openjev-4b ]; then
	fetch openjev-4b AlexWortega/openjev:qwen3.5-4b-nli-v5 a20448012c213128955ca0c693e7c943865cab77 \
		'2fb08b5484dc21cb8e6db9b1fef8437b365f7fa0ef877055cf2bc0f8063a65a2  config.json' \
		'2b5f6e0691a5ed0c5d0f5f3e49bb26abe2d8efd7c14fda4431cd803c12b78f68  model.safetensors' \
		'f399b3cd12fa270d51457bb749fb30863521e8359b8a27059c71b6c2f7d6dd6c  tokenizer.json' \
		'e6f32d049fd16bd0e49604e9173193213fe260984de4629a85d551d85e566020  tokenizer_config.json'
fi

# Inkling 2 Mini: the tokenizer files alone, or with the weights.
inkling2_files='a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2  config.json
a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7  generation_config.json
7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6  tokenizer.json
bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636  tokenizer_config.json
c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0  special_tokens_map.json
53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc  chat_template.jinja'
if [ "$want" = inkling2-mini ]; then
	inkling2_files="$inkling2_files
c1631065b7b00787993522dddc66a68b3abee869192977fb3dcb796e63bc5838  model.safetensors.index.json
65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1  model-00001-of-00001.safetensors"
fi
if [ "$want" = inkling2-mini ] || [ "$want" = inkling2-mini-tokenizer ]; then
	(
		IFS='
'
		# shellcheck disable=SC2086 # one argument per line
		fetch inkling2-mini thinkingmachines/Inkling-2-Mini ed40e7da625f594fc9c4990f5f66be24873c22d7 $inkling2_files
	)
fi

if [ "$want" = yolos-tiny ]; then
	fetch yolos-tiny hustvl/yolos-tiny 95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de \
		'a171607bf33c3c85c2cda36e63f681e8b7fd101396b3f0f7a4aa40214020b86e  config.json' \
		'4309cf0226631c9cd34e6981af842c74f4c9214932741d7fc84b68715c36cf66  preprocessor_config.json' \
		'5a6a017a20cb522dd347271fa5bd670467e456176aaccd940090e50985ac6e74  model.safetensors'
fi

if [ "$want" = yolos-small-300 ]; then
	fetch yolos-small-300 hustvl/yolos-small-300 720d7bcdf9c2028c6c65754012cae4344e9b1f0e \
		'340c26d18da821b80b7e3a558697f0d39328f99f9f4eabe246ea9851e7f50de5  config.json' \
		'9f61f8d9ab3b8ead8918198393053a92f8058d34ec92661b4ee1bc5a2365b9f6  preprocessor_config.json' \
		'ab9cb39a27f9e9c4dc0a9616685304c0b94be0930ddbb695c0a97ca84ec146f9  model.safetensors'
fi
