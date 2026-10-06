<#
  iRent 車況管家 demo 啟動腳本（請用 UTF-8 with BOM 儲存，PowerShell 5.1 才讀得對中文）
    powershell -ExecutionPolicy Bypass -File .\run.ps1                安裝核心套件、產憑證、同時開 http:8000 與 https:8443
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipInstall   已裝過就跳過安裝
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -WithVision    另外安裝 EasyOCR + YOLO（約 1 GB，「自動判讀」劇本才需要）
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -HttpOnly      只開 http（沒有即時取景框）
#>
param([switch]$SkipInstall, [switch]$WithVision, [switch]$HttpOnly, [int]$HttpPort = 8000, [int]$HttpsPort = 8443)
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$env:IRENT_HTTP_PORT = "$HttpPort"; $env:IRENT_HTTPS_PORT = "$HttpsPort"

# 埠被占住（通常是上次沒關乾淨的 python）就直接說，不要讓 uvicorn 啞掉
foreach ($port in @($HttpPort) + $(if (-not $HttpOnly) { @($HttpsPort) } else { @() })) {
  $owner = (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
  if ($owner) { Write-Error "埠 $port 已被 PID $owner 占用。先 taskkill /F /PID $owner，或改用 -HttpPort / -HttpsPort 換埠。"; exit 1 }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Error '找不到 uv。先執行：winget install astral-sh.uv'; exit 1 }
if (-not (Test-Path $py)) { Write-Host '建立 Python 3.12 環境…'; cmd /c 'uv venv .venv --python 3.12' }
if (-not $SkipInstall) {
  Write-Host '安裝核心套件…'; cmd /c 'uv pip install --python .venv\Scripts\python.exe -r requirements.txt'
  if ($WithVision) { Write-Host '安裝 EasyOCR + YOLO（會很久）…'; cmd /c 'uv pip install --python .venv\Scripts\python.exe -r requirements-vision.txt' }
}

# 區網 IP：排除 loopback、APIPA、Tailscale(100.x)、Hyper-V 虛擬網卡，Wi-Fi 優先
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -notlike '100.*' -and $_.InterfaceAlias -notlike 'vEthernet*' } |
  Sort-Object { $_.InterfaceAlias -notlike '*Wi-Fi*' } | Select-Object -First 1).IPAddress
if (-not $ip) { $ip = '127.0.0.1' }
$env:IRENT_LAN_IP = $ip

# 公用網路會擋掉手機的連線
$prof = Get-NetConnectionProfile -ErrorAction SilentlyContinue | Where-Object {
  (Get-NetIPAddress -InterfaceIndex $_.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress -contains $ip }
if ($prof -and $prof.NetworkCategory -eq 'Public') {
  Write-Warning "目前網路「$($prof.Name)」是「公用」網路，手機通常連不進筆電。到 設定 → 網路和網際網路 → Wi-Fi → 該網路 → 網路設定檔類型 改成「私人」。"
}

# 自簽憑證：iOS 要求 EKU serverAuth，IP 變了或版本不同就重產
$certs = Join-Path $root 'certs'; New-Item -ItemType Directory -Force $certs | Out-Null
$certFile = Join-Path $certs 'cert.pem'; $keyFile = Join-Path $certs 'key.pem'; $san = Join-Path $certs 'san.txt'
$openssl = (Get-Command openssl -ErrorAction SilentlyContinue).Source
if (-not $openssl) { $openssl = 'C:\Program Files\Git\usr\bin\openssl.exe' }
$want = "v2|$ip"
if (-not (Test-Path $certFile) -or -not (Test-Path $san) -or ((Get-Content $san -Raw).Trim() -ne $want)) {
  Write-Host "產生自簽憑證（$ip）…"
  if (Test-Path $certFile) { Remove-Item $certFile -Force }
  if (Test-Path $openssl) {
    $env:MSYS_NO_PATHCONV = '1'
    & $openssl req -x509 -newkey rsa:2048 -nodes -keyout $keyFile -out $certFile -days 365 -subj '/CN=irent-demo' `
      -addext "subjectAltName=IP:$ip,IP:127.0.0.1,DNS:localhost" -addext 'extendedKeyUsage=serverAuth' `
      -addext 'basicConstraints=critical,CA:FALSE' -addext 'keyUsage=critical,digitalSignature,keyEncipherment' 2>$null
  }
  if (Test-Path $certFile) { Set-Content $san $want } else { Write-Warning '產生憑證失敗（找不到 openssl？），改成只開 http'; $HttpOnly = $true }
}

Write-Host ''
Write-Host '======================================================'
Write-Host "  總覽（QR code）:    http://localhost:$HttpPort/"
if (-not $HttpOnly) { Write-Host "  手機 取景框版:      https://$ip`:$HttpsPort/app" }
Write-Host "  手機 備用 HTTP 版:  http://$ip`:$HttpPort/app"
Write-Host "  後台工作台:         http://localhost:$HttpPort/admin"
Write-Host '  第一次啟動 Windows 會問是否允許 python 使用網路：請勾「私人」；若目前網路是公用，也要勾「公用」。'
Write-Host '  QR code 請用手機相機或 Chrome 掃，不要用 LINE 掃（LINE 內建瀏覽器無法略過憑證警告）。'
Write-Host '======================================================'
Write-Host ''

$http = Start-Process -FilePath $py -ArgumentList '-m','uvicorn','app.main:app','--host','0.0.0.0','--port',"$HttpPort" -PassThru -NoNewWindow
try {
  if ($HttpOnly) { Wait-Process -Id $http.Id }
  else { & $py -m uvicorn app.main:app --host 0.0.0.0 --port $HttpsPort --ssl-keyfile $keyFile --ssl-certfile $certFile }
} finally {
  if ($http -and -not $http.HasExited) { Stop-Process -Id $http.Id -Force }
}
