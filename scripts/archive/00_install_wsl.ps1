# ============================================================
# STEP 0: Install WSL2 + Ubuntu on Windows
# ============================================================
# Run this in Windows PowerShell (as Administrator)
# Right-click Start → "Terminal (Admin)" or "PowerShell (Admin)"
# ============================================================

Write-Host "═══ Step 0: Installing WSL2 + Ubuntu ═══" -ForegroundColor Cyan

# Enable WSL
wsl --install -d Ubuntu-22.04

Write-Host ""
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Green
Write-Host " WSL2 + Ubuntu 22.04 installation started!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host " WHAT HAPPENS NEXT:" -ForegroundColor Yellow
Write-Host " 1. Your computer will RESTART" -ForegroundColor Yellow
Write-Host " 2. After restart, Ubuntu will open automatically" -ForegroundColor Yellow
Write-Host " 3. It will ask you to create a username + password" -ForegroundColor Yellow
Write-Host "    (this is your Linux username, not Windows)" -ForegroundColor Yellow
Write-Host " 4. After that, open this folder in VSCode:" -ForegroundColor Yellow
Write-Host "    Press Ctrl+Shift+P → 'WSL: Connect to WSL'" -ForegroundColor Yellow
Write-Host " 5. Then run: bash 01_setup_environment.sh" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Green
