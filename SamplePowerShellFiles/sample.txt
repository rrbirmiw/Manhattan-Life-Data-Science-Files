Import-Module PSExcel
Import-Module MergeCSV
Import-Module importexcel
Import-Module ImportExcel

#-----------------------------------------------------------------------------------
# PARAMETERS #
[string]$WorkingDirectory = $PSScriptRoot #get the location of this file 
[string]$pathname = Join-Path  -Path $WorkingDirectory -ChildPath "enrollment.xlsx"
<# 
    Join-Path is a function that takes two arguments. 
        1. -Path is the path of this folder, which we set to $WorkingDirectory 
        2. -ChildPath is the name of the desired file, with file extension 
#>
#-----------------------------------------------------------------------------------

# Get a workbook
Write-Output 'Example 1'
$Workbook = Import-Excel  $pathname
$Workbook | Format-Table
#-----------------------------------------------------------------------------------

# Get a workbook only selecting FirstName, LastName and Major
Write-Output 'Example 2'
$Workbook2 = Import-Excel  $pathname | Select FirstName, LastName, Major #where FirstName, LastName, Major are exact column names in the Excel File 
$Workbook2 | Format-Table
#-----------------------------------------------------------------------------------

# Inner Join enrollment.xlsx with college_locations.xlsx
Write-Output 'Example 3'
$enrollment = Import-Excel -Path (Join-Path -Path $WorkingDirectory -ChildPath "enrollment.xlsx")
<#
    In the above line, we set the "PATH" parameter of the function Import-Excel to the value within the parenthesis 
    Note that the parenthesis is a must due to left-to-right order of operation execution

#>

$locations = Import-Excel -Path (Join-Path -Path $WorkingDirectory -ChildPath "college_locations.xlsx")
$merged = Join-Object -Left $enrollment -Right $locations -LeftJoinProperty "College" -RightJoinProperty "College" -Type  OnlyIfInBoth
$merged | Format-Table
#-----------------------------------------------------------------------------------



# Get only College Name and City and Display
Write-Output 'Example 4'
$merged | Select College, City | Format-Table
#-----------------------------------------------------------------------------------

# Get only UNIQUE College Name and City and Display
Write-Output 'Example 5'
$merged | Select -Unique College, City  | Format-Table
#-----------------------------------------------------------------------------------

# Select Person (firstname & lastname) and their current city
Write-Output 'Example 6'
$livesIn = $merged | Select -Unique FirstName, LastName, City
$livesIn | Format-Table
#-----------------------------------------------------------------------------------

#Export livesIn to a Excel file
$livesIn | Export-Excel -Path (Join-Path -Path $WorkingDirectory -ChildPath "livesIn.xlsx")
#-----------------------------------------------------------------------------------

# Select only rows where city==Cambridge
Write-Output 'Example 7'
$livesIn | Where-Object {$_.City -eq 'Cambridge'} | Format-Table
#-----------------------------------------------------------------------------------
