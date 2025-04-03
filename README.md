# MLOps_Project


## Facial Recognition system for student authentication at NYU

# 
Value proposition: 
A facial recognition system will be used for authentication for exams/tests at NYU. This will act as a 2 factor authentication along with the user’s ID card, ensuring that there is no cheating during the exam.

Status Quo: Currently no attendance system or official system exists for verifying student identities. If any class did decide to take attendance manually, it would be hard to verify and take a lot of time

Business metrics:
Security - Unauthorized students will not be able to enter the testing area.
Efficiency - Manual attendance and student verification is time consuming and will take unnecessary manpower
Scalability - This verification will scale well for Online exams, where faces can be checked at one time, if needed to explore in the future

## Contributors

#Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Megh Panandikar                 |                 |                                    |
| Moulik Shah                     |                 |                                    |
| Aryan Tomar                     |                 |                                    |
| Yash Amin                       |                 |                                    |


## System diagram

#Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

## Summary of outside materials

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| MS1MV2       |  	Cleaned and refined MS-Celeb-1M dataset (~5.8M images, 85K IDs)                  |       Free for use           |
| VGGFace2     |    Collected via Google Images; 3.3M images, 9K+ identities                |         Research only          |
| iResNet-50   |    Deep residual network trained on large face datasets               |          Open-source         |
| MTCNN        |    Trained on various public datasets, by MIT                |         Open source         |


## Summary of infrastructure requirements

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| 'compute_skylake' | 2 for entire project                   | 1 in model serving and 1 for model evaluation and load testing           |
| `gpu_v100`     | 4 gpus/ 4 slots of 5 hours                        |       Required for training the ResNet-50 for the large database        |
| Floating IPs    | 2 running perpetually |               |    1 for model serving api, 1 for monitoring while training, testing and serving
| Persistent Volume  - 'block storage'   |                1 volume - 10GB                                  |       Needed to store model checkpoints, logs, retrained models, and OpenVINO outputs        |
| Object Storage 'CHI@TACC Swift' |   1 volume - 50GB     |  Storing the static dataset

## Detailed design plan

#In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

### Model training and training platforms

#Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

### Model serving and monitoring platforms

#Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

### Data pipeline

#Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

### Continuous X

#Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. --> 

