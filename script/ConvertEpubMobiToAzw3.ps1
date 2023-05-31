Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
# change follow path if your calibre is installed in a different path
$exe = "C:\Program Files\Calibre2\ebook-convert.exe"
$extList = @(
    '.mobi'
    '.epub'
)
$targetExt = ".azw3"

function Update-String {

    param (
        [string]$InputString,
        [string[]]$ReplaceStringList,
        [string]$TargetString
    )

    $newString = $InputString

    foreach($str in $ReplaceStringList.GetEnumerator())
    {
        $newString = $newString -replace $str, $TargetString
    }

    return $newString
}

$files = Get-ChildItem -File | Where-Object {($_.Extension -eq ".epub") -or ($_.Extension -eq ".mobi")}
ForEach ($file in $files)
{
  $inputName = '"' + $file.FullName + '"'
  $outputName = Update-String -InputString $inputName -ReplaceStringList $extList -TargetString $targetExt
  if ($file.FullName -ne $newName)
  {
      & $exe $inputName $outputName
  }
}
