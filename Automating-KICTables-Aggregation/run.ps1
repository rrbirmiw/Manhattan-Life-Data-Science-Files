

<# This powershell script takes ALL .xlsx files in the directory (WORKING_DIRECTORY)
(see below) -- as well as ALL subfolders, etc (recursively) and takes the matrix of
data, transposes it, and appends them in order to the file designated in
OUTPUT_FILENAME. The one exception is THE OUTPUT FILE (named, OUTPUT_FILENAME, see user parameters)
The user may control the number of files to process using the "N" user parameter below

USER PARAMETERS
  j0: assumed = 5, is the starting row of data in each of the _source_ files
  i0: assumed = 2, likewise, starting column
  y0: assumed = 2. row in the SUMMARY (output) file where want first set of data to go
  x0: assumed = 3 likewise, column
  N: assumed = 99999 (a very large value): number of source files to process
  spacing: assumed = 3. number of rows padded in between the data regions of two consecutive files
  nums_col: assumed = 1: column where the numbers 1.....N, denoting the row number of the matrices, goes

 Written By Rahul Birmiwal 2018
#>

######################## USER PARAMETERS #####################################################

#[string]$Working_Directory = 'C:\Users\Rahulbirmiwal\Documents\KIC TABLES' #uncomment this line to manually set working directory
[string]$Working_Directory = Get-Location
[string]$OUTPUT_FILENAME = "MasterSummary.xlsx"
[string]$outfile_path = Join-Path -Path $Working_Directory -ChildPath $OUTPUT_FILENAME

#User-Display Parameters
[int]$j0 = 5 #starting row of data in each of the KIC_ files
[int]$i0 = 2 #starting col of data in each of the KIC_files
[int]$y0 = 2 #row where you want data to start in the SUMMARY (this) file
[int]$x0 = 3 #column where want data to start in the SUMMARY (This) file
[int]$spacing = 3
[int]$nums_col = 2
[int]$N = 999 #for testing

####################### HELPER FUNCTIONS  #######################################

Function Convert-NumberToA1 {
  <# https://gallery.technet.microsoft.com/office/Powershell-function-that-88f9f690
  .SYNOPSIS
  This converts any integer into A1 format.
  .DESCRIPTION
  See synopsis.
  .PARAMETER number
  Any number between 1 and 2147483647
  #>

  Param([parameter(Mandatory=$true)]
        [int]$number)

  $a1Value = $null
  While ($number -gt 0) {
    $multiplier = [int][system.math]::Floor(($number / 26))
    $charNumber = $number - ($multiplier * 26)
    If ($charNumber -eq 0) { $multiplier-- ; $charNumber = 26 }
    $a1Value = [char]($charNumber + 64) + $a1Value
    $number = $multiplier
  }
  Return $a1Value
}


#https://stackoverflow.com/questions/10928030/in-powershell-how-can-i-test-if-a-variable-holds-a-numeric-value
function Is-Numeric ($Value) {
    return $Value -match "^[\d\.]+$"
}


### #Get a list of files to copy from
$Files = GCI $Working_Directory -Recurse | ?{$_.Extension -Match "xlsx?"} | select -ExpandProperty FullName

#Launch Excel
$Excel = New-Object -ComObject Excel.Application
$Excel.Visible = $False
$Excel.DisplayAlerts = $False



### create output file
$output_wb = $Excel.Workbooks.Add()
$output_wksheet = $output_wb.worksheets.item(1)

### conversion from numeric to A1 excel range format
[int]$output_paste_cell_row = 3
[string]$output_paste_cell_col = Convert-NumberToA1 -number $x0
[string]$output_nums_col = Convert-NumberToA1 -number $nums_col
$output_paste_cell = $output_paste_cell_col + $output_paste_cell_row #initial cell to paste transposed KIC matrix


write-output "N is " + $N
###################### MAIN PROCESSING LOOP #######################################################
[int]$counter = 0
ForEach($File in $Files) {

      $counter++
      if ($counter -ge $N ) { break}

      Write-Output $File

      [string]$extracted_fname = $File.split("{\}")[-1] #get only relevant portion of filename
      write-output $extracted_fname

      if ($extracted_fname -eq $OUTPUT_FILENAME) {continue}

      ## Get source and range parameters
      $Source = $Excel.Workbooks.Open($File,$null,$true)
      [int]$lastColValue = $Source.Worksheets.item(1).UsedRange.rows.count
      [int]$lastRowvalue = $Source.Worksheets.item(1).UsedRange.columns.count
      [string]$lastCol_letter = Convert-NumberToA1 -number $lastColValue
      [string]$lastRow_letter = Convert-NumberToA1 -number $lastRowvalue
      [string]$endpoint_cell_name = "$" + $lastCol_letter + $lastRowvalue

      ## Copy from source
      $source_wksheet = $source.worksheets.item(1)
      $copy_range = $source_wksheet.range("B5:" + $endpoint_cell_name)
      $copy_range.copy()


      ## Paste transposed KIC matrix
      $output_wksheet.range($output_paste_cell).pasteSpecial(-4163)

      ## Update Destination Cell
      [int]$prev_row_start = $output_paste_cell_row #temp variable
      $output_paste_cell_row +=  $copy_range.columns.count
      $output_paste_cell = $output_paste_cell_col + $output_paste_cell_row #update destination cell

      ### Write Filename to FILENAME_COLUMN
      [string]$filename_range_str = "A"+$prev_row_start + ":" + "A" + $output_paste_cell_row
      $output_wksheet.range($filename_range_str).value2 = $extracted_fname

      ### Write array of row numbers 1....sizeof(KIC Matrix)
      for ($i=$prev_row_start; $i -le ($output_paste_cell_row - $spacing); $i++) {
        $output_wksheet.cells.item( $i, $nums_col).value2 = [int]($i - $prev_row_start + 1)
      }

      ##add spacing
      $output_paste_cell_row += $spacing
      $output_paste_cell = $output_paste_cell_col + $output_paste_cell_row #update destination cell

      #close src
      $source.close($false)

}


############# SAVE OUTPUT ############################################################
$output_wb.saveas($outfile_path)
$Excel.quit()
Return
