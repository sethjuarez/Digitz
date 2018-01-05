$dir = "LearnDigitz\model"
$latest = Get-ChildItem -Recurse -Path $dir | Sort-Object LastAccessTime -Descending | Select-Object -First 1
$latest.FullName
Copy-Item $latest.FullName ".\Digitz"