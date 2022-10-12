---
title: Basic Linux Commands
categories: [Linux, Basic Linux Commands]
tags: [python]     # TAG names should always be lowercase
author: 'Babin'
pin: true
---

- Inside the linux terminal:-
    - the `$` represents the normal user
    - the `#` represents the root or admin user

- To check the hostname we can use the `hostname` command
    For Eg:- 
    
        ```
        Input: hostname
        Output: {returns the hostname}
        ```

- To check the current logged in user we can use the `whoami` command
    For Eg:-

        ```
        Input: whoami
        Output: {returns the current logged in user}
        ```

- To check the IP on Linux we can use the `ip addr` command
    > IP address is an unique address that identifies a device on the internet or a local network
    {: .prompt-info}

    For Eg:-

        ```
        Input: ip addr   
        Output: {returns the ip address} 
        ```

- To print out the working directory we can use the `pwd` command
    For Eg:-

        ```
        Input: pwd
        Output: {prints the working directory}
        ```

- To make a folder on linux we can use the `mkdir` command
    For Eg:- 
        
        ```
        Input: mkdir {directory_name}
        ```

- To change location or to move to another directory we can use the `cd` command
    For Eg:-

        ```
        Input: cd {directory_name}
        ```

- To move back to another directory we can use the `cd` command
    For Eg:-

        ```
        Input: cd ..
        ```
    
    Here, the `..` means that we should move one step back.

- To clear the screen we use the `clear` command
    For Eg:-

        ```
        Input: clear
        ```

- To search for our folder/file inside a specific location we can use the `find`command
    - To search for folder
        For Eg:-

            ```
            Input: find path -name {folder/directory name}
            ```
    - To search for file
        For Eg:-

            ```
            Input: find . -type f -name {filename}
            ```
    
    > We can also use the `locate` command for finding files or folders
    {: .prompt-tip}

- To create a file we can use the `touch` command
    For Eg:-

        ```
        Input: touch {file_name}
        ```

- Removing a directory
    - To remove or delete a directory we can use the `rmdir` command
        For Eg:- 
        
        ```
        Input: rmdir {directory_name}
        ```

    - To remove the directory and all the other files inside of it.
        For Eg:-

        ```
        Input: rm -r {directory_name}
        ```
    
- To view more information about the files we can use the `ls -ltr` command
    For Eg:-

    ```
    Input: ls -ltr
    ```
    Here, the `-l` means we use a long listing format. `-t` means we sort by time, newest first, and `-r` means in the reverse order while sorting. 

    ![img](/assets/img/readwrite.png)

- To view more information about the command we can use either the `man` command or the `--help` arguement
    For Eg:-
    
    ```
    Input: ls --help 
            OR
    Input: man ls
    ```

- To edit or write into a file we can use the  `vi` editor.
    For Eg:-

    ```
    Input: vi {file_name} 
    ```
    To start editing we need to go to `insert` mode for which we must press `i` key. Then we can insert the text as we like. To escape from the insert mode we can press the `escape` key. Now, to save the file, we can press `:wq` where wq means save and quit. 


- To print the file content into the shell we use `cat` command.
    For Eg:-

    ```
    Input: cat {file_name}
    ```

- To count the no.of words and lines we use the `wc` command.
    - To count only the no of lines we use `wc -l`
        For Eg:-

        ```
        Input: wc -l {file_name}
        ```


- To compare two files we use the `diff` command.
    For Eg:-

        ```
        Input: diff {file_1} {file_2}
        ```

- To compree and decompress files
    We use the `tar` do to the packaging of files.
    For Eg:-

        ```
        Input: tar {options} {tar_file_name} {file_1} {file_2}
        Eg: tar cvf files.tar file1 file2
        Output: files.tar
        ```
    Now, we need to compress the files.tar using the `gzip` command 
    For Eg:-

        ```
        Input: gzip {tar_file}
        Eg:- gzip files.tar
        Output:  files.tar.gz
        ```

    Now, to decompress the compresseed file we use the `gunzip` command
    For Eg:-
        
        ```
        Input: gunzip {zipped_file}
        Eg:- gunzip gzip files.tar.gz
        Output: files.tar
        ```
    And lastly, we need to untar the tar files
    For Eg:-

        ```
        Input tar xvf {tar_file}
        Eg:- tar xvf files.tar
        Output: file1, file2s
        ```

- To copy file rom one folder to another we use the `cp` command.
    For Eg:-

    ```
    Input: cp {source_file} {destination_path}
    Eg:- cp files.tar.gz  folder1/
    ```

- To rename a file we use the `mv` command
    For Eg:-

        ```
        Input: mv {old_file_name} {new_file_name}
        ```
        Here, what we do is move the contents of the old file to the new file with a new name and then delete the old file. In linux, we use such indirect renaming of a file. 


- To split and combine the files
    - To combine a file
        For Eg:-
            Let's suppose we want to create a new file- `filec` with the contents of two different files: `filea` and `fileb`. To do this we use the `>` operator.
            For Example:-

                ```
                Input: cat filea fileb > filec
                ```
    - To split a file
        For Eg:- 
            Let's suppose we want to split the content of a `filea` into two files `fileb` and `filec` then we use the `split` command.
            For Example:-

            ```
            Input: split -l 1 filea
            ```
    
- To search for words in a file and show them in a console we use the `grep` command
    For Example:-

    ```
    Input cat {file_name} | grep {word}
    ```
    Here, the `|` is the pipe operator which is used to chain the operations. Here, the 'cat {file_name}' returns some output, and its output is sent as an input to the 'grep {word}` command.

- To read the start and end of the files we use the `head` and `tail` command respectively
    For Example:-
    
        ```
        Input: head -2 {filename}
        Output: prints the first two line from the file. 
        Input: tail -2 {filename}
        Output: prints the last two line from the file.
        ```

- To sort the file we use the `sort` command
    For Example:-

    ```
    Input: sort {file_name}
    ```

- To prin only the unique value we use the `uniq` command
    For Example:-

    ```
    Input: sort {file_name} | uniq
    ```
