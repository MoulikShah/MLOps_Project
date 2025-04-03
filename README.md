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

#Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

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

