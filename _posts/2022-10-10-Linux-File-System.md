---
title: Linux File System
categories: [Linux, Basic Linux Commands]
tags: [python]     # TAG names should always be lowercase
author: 'Babin'
pin: true
---

## Everything in Linux is a File
    Even the basic commands that we use are stored as a file in the linux system. We can see where those files are located using the `whereis` command:

    For eg:- 
    ```
    Input: whereis ls
    Output: ls: /usr/bin/ls /usr/share/man/man1/ls.1.gz
    ```
    This means that the `ls` command is stored as a file in location `/urs/bin`

## File Structure Description
- `/boot` = Having files used by boot loader (ex:grub)
- `/dev` = System devices files (ex: speakers, keyboard etc.)
- `/etc` = Has configuration files
- `/usr/bin` = Binaries
- `/usr/sbin` = System binaries of the root directory
- `/opt` = Installation of optional add-on applications (third party applications)
- `/proc` = Running process
- `/usr/lib` = C Program library files needed by commands and apps
- `/tmp` = Having temporary files
- `/home` = Directories of users
- `/root` = Home directory of root user
- `/var` = System logs
- `/run` = System daemons that start very early (ex: systemd and udev) to store temporary runtime files like PID
- `/mnt` = To mount external filesystem (ex: NFS)
- `/media` = For CDROM Mounts