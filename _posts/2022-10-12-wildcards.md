---
title: Wild Cards Tutorial
categories: [Linux, Basic Linux Commands]
tags: [python]     # TAG names should always be lowercase
author: 'Babin'
pin: true
---
Wildcards are characters that can be used as a substitute for any of a class of characters in a search. 
Suppose there are hundreds and thousands of files and we have to find only xml files then in such cases we can use wild cards. 
```
    Input: ls -l *.xml
    Output: Lists all files with "xml" extension
```

Some common wild cards are:-
- `*`: zero or more characters
- `?`: single character
- `[]`: range of character
- `^`: beginning of the line
- `$`: end of the line
  
### Suppose we want to create 20 different files with names file1.txt file2.txt ...file20.txt
```
    Input: touch file{1..20}.txt
    Output: Creates 20 different files with names file1.txt, file2.txt upto file20.txt
```

### Removing all files with filenames file1..file20.txt
```
    Input: rm file*.txt
    Output: Removes all files with extension .txt that begins with string `file`.
```

### Suppose we have to list every files in which first character can be anything but must be followed by 123
```
    Input: ls -l ?123
    Output: Lists every file whose first character is followed by 123
```

### Suppose we have to list all files that starts with `te` and ends with `t` but has a random character in between
```
    Input: ls -l te?t
    Output: Lists every files that starst with te and ends with t with a random character in between
```

### Suppose we have to list all files that starts with either a or b or c followed by 123
```
    Input: ls -l [abc]123
```

### Suppose we have to list all files that starts with either any alphabet followed by 123
```
    Input: ls -l [a-z]123
```

### Suppose we have to list all files that have a number in its filename
```
    Input: ls -l *[0-9]*
```

### Suppose we have a file names called `names` that has a bunch of name it it and want to print every line that start with R
```
    #names
    Leonard
    Raj
    Sheldon
    Howard
    Bernadette
    Amy
    Penny
``` 

```
    Input: cat names | grep ^R
    Output: Raj
```

### Suppose we have a file names called `names` that has a bunch of name it it and want to print every line that ends with y```
    #names
    Leonard
    Raj
    Sheldon
    Howard
    Bernadette
    Amy
    Penny
``` 

```
    Input: cat names | grep y$
    Output: Raj
```