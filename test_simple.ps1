# Simple Feature Test Script
Write-Host "`n=== Testing Noah's AI Assistant Features ===" -ForegroundColor Cyan

$apiUrl = "https://noahsaiassistant.vercel.app/api/chat"

# Test 1: System Architecture
Write-Host "`n[TEST 1] System Architecture Question..." -ForegroundColor Yellow

$body = @{
    query = "Show me the complete system architecture"
    role = "Software Developer"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body -ContentType "application/json"

Write-Host "Response length: $($response.answer.Length) chars" -ForegroundColor Green
Write-Host "Has junior content: $(($response.answer -match 'junior' -or $response.answer -match 'simple'))" -ForegroundColor Green
Write-Host "Has senior content: $(($response.answer -match 'senior' -or $response.answer -match 'advanced'))" -ForegroundColor Green
Write-Host "Has code blocks: $($response.answer -match '```')" -ForegroundColor Green

Write-Host "`nPreview:" -ForegroundColor Cyan
Write-Host $response.answer.Substring(0, 300)
Write-Host "..."

Start-Sleep -Seconds 3

# Test 2: Data Analytics
Write-Host "`n[TEST 2] Data Analytics Display..." -ForegroundColor Yellow

$body2 = @{
    query = "Can you display data collected?"
    role = "Hiring Manager (technical)"
} | ConvertTo-Json

$response2 = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body2 -ContentType "application/json"

Write-Host "Response length: $($response2.answer.Length) chars" -ForegroundColor Green
Write-Host "Has tables: $(($response2.answer -match '\|' -or $response2.answer -match 'Table'))" -ForegroundColor Green
Write-Host "Has metrics: $(($response2.answer -match 'Average' -or $response2.answer -match 'Total'))" -ForegroundColor Green

Write-Host "`nPreview:" -ForegroundColor Cyan
Write-Host $response2.answer.Substring(0, 300)
Write-Host "..."

Start-Sleep -Seconds 3

# Test 3: Follow-up Suggestions
Write-Host "`n[TEST 3] Multi-Choice Follow-ups..." -ForegroundColor Yellow

$body3 = @{
    query = "How does this product work?"
    role = "Software Developer"
} | ConvertTo-Json

$response3 = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body3 -ContentType "application/json"

Write-Host "Response length: $($response3.answer.Length) chars" -ForegroundColor Green
Write-Host "Has follow-ups: $(($response3.answer -match 'Explore' -or $response3.answer -match 'next'))" -ForegroundColor Green
Write-Host "Has bullet points: $($response3.answer -match '-')" -ForegroundColor Green

Write-Host "`nPreview:" -ForegroundColor Cyan
Write-Host $response3.answer.Substring(0, 300)
Write-Host "..."

Write-Host "`n=== All Tests Complete! ===" -ForegroundColor Green
Write-Host "Visit: https://noahsaiassistant.vercel.app" -ForegroundColor Cyan
