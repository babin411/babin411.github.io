---
title: Process Management in Linux
categories: [Linux, Basic Linux Commands]
tags: [python]     # TAG names should always be lowercase
author: 'Babin'
pin: true
---

## Some process management commands in Linux
- `jobs`: Will show active jobs
- `bg`: Resume jobs to the background
- `fg`: Resume job to the foreground
  
- To resume a speciic jobs 
    ```
        Syntax: 
            bg %<job_id> (To resume the job in backround)
            gh %<job_id> (To resume the job in foreground)

    ```

## Nice value
Every process has a nice value which range goes from -20 to 10. The lower the value is, the more priority that the process gets. 
    
    ```
        # To check the nice value of a process
        Syntax: 
            ps -l <PID>

        # To change the priority of a process by chaning its nice value
        Syntax:
            ps -n <nice vlaue> <PID>
    ```

## nohup
If we want our process to keep running even after closing our terminal, we can use nohup.

    ```
        Syntax:
            nohup process &
    ```