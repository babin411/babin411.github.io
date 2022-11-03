---
title: Linux Redirects
categories: [Linux, Basic Linux Commands]
tags: [python]     # TAG names should always be lowercase
author: 'Babin'
pin: true
---

There are 3 redirects: <br/>
Basically, when we run a command in terminal, belwo three files are created.

| Company                      | File Descriptor  |
|:-----------------------------|:----------------:|
| Standard Input (stdin)| 0|
| Standard Output (stdout)| 1|
| Standard Error (stderr)| 2|

## Output (stdout-1)
- Output of a command is shown in terminal.
- To route output in file using `>`. This clears all the previous content and then overwrites with the new content. 
  - `hostname` > file_name
- To append output in existing file using `>>`. This appends the new content to the previous existing content.
  - `pwd` >> file_name

## Error (stderr - 2)
- If any command gives your error then it is considered as `stderr-2`
- We can redirect the error to a file:
  - Eg: cd /root/ 2>error_file
  - This prevents the error from showing up in the terminal but writes all the error in a file. 
- To redirect both standard output and error to a file:
  Eg:-
  
  ```
    Input: hostname >> std_err_out 2>&1
    Input: cat std_err_out
    Output: <Host Name of the Computer>

    Input: cd /root/  >> std_err_out 2>&1
    Input cat std_err_out
    Output: 
        <Host Name Of The Computer>
        <Error>
  ```

## Input (stdin - 0)
- Input is used when feeding file contents to a file
- Eg:- 
  
  ```
    cat < file_name
    cat << EOF
  ```