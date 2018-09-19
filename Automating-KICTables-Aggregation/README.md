# File Formats: 

execute_run_txt.txt -> Save As -> File Type --> All File Types --> Save with name 'execute_run.bat'
run_txt.txt -> Save As -> File Type --> All File Types --> Save with name 'run.ps1'

# To Use: 

Double click 'execute_run', which is a windows batch file that will run the Powershell script run.ps1 
This will create the master/summary worksheet named "MasterSummary"
By default this code will only aggregate N=6 total files (for testing purposes)
**The code will run on ALL .xlsx files in the working directory, and in all subfolders, recursively**
**Therefore please make sure run.ps1 and execute_run.bat are present in a 'cleaned-out' folder (containing only the source files needed) 

Additional document is in the source code file. Rightclick run.ps1 -> open with Notepad, etc. 
To edit the code, I recommend using Windows Powershell ISE, which is the native editor (Windows Button -> Search for Windows Powershell ISE)
another option is to download more advanced editors such as Atom or Sublime Text 

With that said, to run the full version, please open code, go to section "USER PARAMETERS" and set N to a high number, like 99999
Also, if one would like to change the name of the output file, analogous to above, set $OUTPUT_FILENAME = <> to your desired name in 
"my_name.xlsx" format (quotation marks required; no full filepath needed) 


Thanks

Rahul Birmiwal 2018
