# Изтегля AI моделите в tools\, ако липсват или са повредени. Зарежда се от setup-windows.ps1 и
# install-task.ps1, така че обновяването на сървъра изтегля и моделите, нужни на по-нова версия.
$ProgressPreference = 'SilentlyContinue' # лентата за напредък прави Invoke-WebRequest около 10 пъти по-бавен
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$SEG_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx'
$SEG_SHA256 = '60920E99C45464F2BA57BEE2AD08C919A52BBF852739E96947FBB4358C0D964A'
$DEPTH_URL = 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx'
$DEPTH_SHA256 = 'AFB6A5C28F3B6BF1618C6E43F02073EF9DFDC70E937502D51603E57B0A1DF10C'
$NORMAL_URL = 'https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx'
$NORMAL_SHA256 = '24EACB5DC7A2C54C7BC98F7DE085FFBED79AD006EA5B664C2C2CDC02FF3A52F0'
$SAM_BASE = 'https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8'
$SAM_ENCODER_SHA256 = '580F5FB648EA1062C0AABC26217AED56921985F03F0CBBD852BBA81D760CC749'
$SAM_DECODER_SHA256 = '8976B90A87BA50A6A72217A5FF994F7D25CE16F2229FCC1ED259E1294C622FFE'
# Безплатният AI модел на Pro от един кадър (app/single3d.py): теглата и кодът на TripoSR, с фиксирана версия.
$TRIPOSR_BASE = 'https://huggingface.co/stabilityai/TripoSR/resolve/5b521936b01fbe1890f6f9baed0254ab6351c04a'
$TRIPOSR_CKPT_SHA256 = '429E2C6B22A0923967459DE24D67F05962B235F79CDE6B032AA7ED2FFCD970EE'
$TRIPOSR_CONFIG_SHA256 = '74CA708CE086BF68E97709EA6B3D91F14717921C04691E84043F0EB8FCC68E62'
$TRIPOSR_COMMIT = '107cefdc244c39106fa830359024f6a2f1c78871'
# Текстовите промени на Pro (app/edit_llm.py): Qwen2.5-1.5B-Instruct, с фиксирана версия; име на файл -> SHA-256.
$QWEN_BASE = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
$QWEN_FILES = [ordered]@{
  'config.json' = '98D2FF8CC47488D08A2B0B3ACF4EB99EF210779B42BD48605F6B8E36ACDBF670'
  'generation_config.json' = 'E558847A8B4402616F1273797B015104DC266FE4B520056FCA88823BA8F8EBE6'
  'merges.txt' = '599BAB54075088774B1733FDE865D5BD747CBCC7A547C5BC12610E874E26F5E3'
  'tokenizer.json' = 'C0382117EA329CDF097041132F6D735924B697924D6F6FC3945713E96CE87539'
  'tokenizer_config.json' = '5B5D4F65D0ACD3B2D56A35B56D374A36CBC1C8FA5CF3B3FEBBBFABF22F359583'
  'vocab.json' = 'CA10D7E9FB3ED18575DD1E277A2579C16D108E32F27439684AFA0E10B1440910'
  'model.safetensors' = 'DD924A11B4C220F385B51FFA522DAEA7C9F3D850E31B162BB5661DF483C6D3EE'
}
# Промените на цвета на Pro (app/recolor.py): CLIPSeg (CIDAS/clipseg-rd64-refined, Apache-2.0) като ONNX, с фиксирана версия.
$CLIPSEG_BASE = 'https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/924dc94f85f58739f353f94258b33bc47eae4862'
$CLIPSEG_MODEL_SHA256 = '25C50E9A236FA16F309AAFD54D363ADB7CEE9CC7EAD8D93CF31A2610F8AF4B37'
$CLIPSEG_TOKENIZER_SHA256 = '342739D4A4B111D4B1BDA83A96F5C518E1B162CC597252AC89DED17681791D31'
# Какъв е предметът на една снимка (app/kind.py): CLIP (openai/clip-vit-base-patch32, MIT) като ONNX, с фиксирана версия.
$CLIP_BASE = 'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/6b7c523703ae26e39be4b03774966f8571c7f523'
$CLIP_MODEL_SHA256 = '0898A3FACFDB27F0A041E57649B4989CFD094E4A0040D6AE75ED69917DFC7328'
$CLIP_TOKENIZER_SHA256 = 'F7F3B7AF117D467B58374797691A6438D3E6B9E9CEF800DFD5DCED7F697A90CD'
$modelDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'tools'

function Install-TripoSRCode {
  $code = Join-Path $modelDir 'triposr\code'
  $mark = Join-Path $code "commit-$TRIPOSR_COMMIT"
  if (Test-Path $mark) { Write-Host 'Already installed'; return }
  $zip = Join-Path $modelDir 'triposr\code.zip'
  New-Item -ItemType Directory -Force (Split-Path $zip) | Out-Null
  Write-Host "Downloading TripoSR code ($TRIPOSR_COMMIT)"
  Invoke-WebRequest "https://github.com/VAST-AI-Research/TripoSR/archive/$TRIPOSR_COMMIT.zip" -OutFile $zip -UseBasicParsing
  $tmp = Join-Path $modelDir 'triposr\code.tmp'
  Remove-Item $tmp, $code -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive $zip $tmp -Force
  Move-Item (Join-Path $tmp "TripoSR-$TRIPOSR_COMMIT") $code
  Remove-Item $tmp, $zip -Recurse -Force
  if (-not (Test-Path (Join-Path $code 'tsr\system.py'))) { throw 'The TripoSR code download is incomplete. Run this again.' }
  New-Item -ItemType File $mark | Out-Null
  Write-Host 'OK'
}

function Install-Model($file, $url, $sha256) {
  if ((Test-Path $file) -and (Get-FileHash $file -Algorithm SHA256).Hash -eq $sha256) { Write-Host 'Already installed'; return }
  New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
  Write-Host "Downloading $url"
  Invoke-WebRequest $url -OutFile "$file.part" -UseBasicParsing
  if ((Get-FileHash "$file.part" -Algorithm SHA256).Hash -ne $sha256) { throw 'The model download is damaged. Run this again.' }
  Move-Item "$file.part" $file -Force
  Write-Host 'OK'
}

function Install-Models {
  Write-Host "`n==> AI object finder (180 MB, tells the object apart from the background)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'seg\isnet-general-use.onnx') $SEG_URL $SEG_SHA256
  Write-Host "`n==> AI depth model (100 MB, Depth Anything V2 Small: the shape of plain and shiny surfaces)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'depth\depth_anything_v2_small.onnx') $DEPTH_URL $DEPTH_SHA256
  Write-Host "`n==> AI fine detail (141 MB, MoGe-2 small: the direction of every part of the surface)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'depth\moge2_vits_normal.onnx') $NORMAL_URL $NORMAL_SHA256
  Write-Host "`n==> AI outline sharpener (45 MB, MobileSAM: cuts the object out where the 3D shape points)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'sam\mobile_sam_image_encoder.onnx') "$SAM_BASE/mobile_sam_image_encoder.onnx" $SAM_ENCODER_SHA256
  Install-Model (Join-Path $modelDir 'sam\sam_mask_decoder_multi.onnx') "$SAM_BASE/sam_mask_decoder_multi.onnx" $SAM_DECODER_SHA256
  Write-Host "`n==> Pro's AI from one picture (1.7 GB, TripoSR: a 3D model from a single frame, free)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'triposr\weights\config.yaml') "$TRIPOSR_BASE/config.yaml" $TRIPOSR_CONFIG_SHA256
  Install-Model (Join-Path $modelDir 'triposr\weights\model.ckpt') "$TRIPOSR_BASE/model.ckpt" $TRIPOSR_CKPT_SHA256
  Install-TripoSRCode
  Write-Host "`n==> Pro's text changes (3.1 GB, Qwen2.5-1.5B-Instruct: reads what the user wants changed, free)" -ForegroundColor Cyan
  foreach ($f in $QWEN_FILES.Keys) { Install-Model (Join-Path $modelDir "qwen\$f") "$QWEN_BASE/$f" $QWEN_FILES[$f] }
  Write-Host "`n==> Pro's colour changes (550 MB, CLIPSeg: finds the part to recolour, e.g. the hair, free)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'clipseg\model.onnx') "$CLIPSEG_BASE/onnx/model.onnx" $CLIPSEG_MODEL_SHA256
  Install-Model (Join-Path $modelDir 'clipseg\tokenizer.json') "$CLIPSEG_BASE/tokenizer.json" $CLIPSEG_TOKENIZER_SHA256
  Write-Host "`n==> AI that names the object in a photo (155 MB, CLIP: a cup is made round, a remote flat, free)" -ForegroundColor Cyan
  Install-Model (Join-Path $modelDir 'clip\model_quantized.onnx') "$CLIP_BASE/onnx/model_quantized.onnx" $CLIP_MODEL_SHA256
  Install-Model (Join-Path $modelDir 'clip\tokenizer.json') "$CLIP_BASE/tokenizer.json" $CLIP_TOKENIZER_SHA256
}
