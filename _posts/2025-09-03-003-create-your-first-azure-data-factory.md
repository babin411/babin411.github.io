---
title: "Create your First Azure Data Factory"
categories: [Cloud, Azure, Azure Data Factory]
tags: [azure, azure data factory, adf, data engineering]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/adf/notion/page-003/image-01.png
  alt: Create a first Azure Data Factory from Notion notes
---

## Reference

[Bookmark from the original Notion lesson](https://app.notion.com/p/0e9b4e7280954dc3a2a94626a4cf4e50#ef1bac8dc93d4c21b9ffd203038f9935)

Overview: Want to copy a file from one folder to another folder using copy activity in ADF.

1. Create Storage Account and a Container

   ![Create Storage Account and a Container](/assets/img/adf/notion/page-003/image-01.png)

2. Create an Input Folder and place a file that we want to copy.

   ![Create an input folder and place a file](/assets/img/adf/notion/page-003/image-02.png)

3. Create a Data Factory

   ![Create a Data Factory](/assets/img/adf/notion/page-003/image-03.png)

4. Go to ADF and goto Author and Monitor

   ![Go to ADF and Author and Monitor](/assets/img/adf/notion/page-003/image-04.png)

5. Create a Linked Service: Since for our demo, our ADF needs to interact with an external resource which in our case is the storage account we have to create a linked service.

   ![Create a linked service](/assets/img/adf/notion/page-003/image-05.png)

6. Create a Dataset for source and sink

   1. Source Dataset

      ![Create source dataset](/assets/img/adf/notion/page-003/image-06.png)

      ![Configure source dataset](/assets/img/adf/notion/page-003/image-07.png)

      ![Source dataset file settings](/assets/img/adf/notion/page-003/image-08.png)

   2. Sink Dataset

      ![Create sink dataset](/assets/img/adf/notion/page-003/image-09.png)

7. Create a Pipeline

   ![Create a pipeline](/assets/img/adf/notion/page-003/image-10.png)

   ![Configure copy activity pipeline](/assets/img/adf/notion/page-003/image-11.png)

   ![Pipeline copy activity settings](/assets/img/adf/notion/page-003/image-12.png)

8. Validate your pipeline. Incase we forget to configure anything in our pipeline then it will notify us.

9. Publish our pipeline. Once we publish, all the pipelines, linked services, datasets and triggers will be published in our ADF.

10. Incase we just want to check if our pipeline is running as intended for debugging purpose before publishing we can use the `Debug` mode.

## Source

Based on my Notion lesson page: [3. Create your First Azure Data Factory](https://app.notion.com/p/0e9b4e7280954dc3a2a94626a4cf4e50).
