# MLOps_Project


## Facial Recognition system for student authentication at NYU

# 
Value proposition: 
A facial recognition system will be used for authentication for exams/tests at NYU. This will act as a 2 factor authentication along with the user’s ID card, ensuring that there is no cheating during the exam.

Status Quo: Currently no attendance system or official system exists for verifying student identities. If any class did decide to take attendance manually, it would be hard to verify and take a lot of time

Business metrics:
- Security: Unauthorized students will not be able to enter the testing area.
- Efficiency: Manual attendance and student verification is time consuming and will take unnecessary manpower
- Scalability: This verification will scale well for Online exams, where faces can be checked at one time, if needed to explore in the future

## Contributors

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |   System diagram, Planning                   |                                                                              |
| Megh Panandikar                 |   Model serving  and Monitoring              |  https://github.com/MoulikShah/MLOps_Project/commits/main/?author=megh2001   |
| Moulik Shah                     |   Data pipeline                              |  https://github.com/MoulikShah/MLOps_Project/commits/main/?author=MoulikShah |
| Aryan Tomar                     |   Model training                             |  https://github.com/MoulikShah/MLOps_Project/commits/main/?author=aryntmr    |
| Yash Amin                       |   Continuous X pipeline                      |  https://github.com/MoulikShah/MLOps_Project/commits/main/?author=Yash-5865  |


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

Persistent Storage

- We provision block storage volumes on **Chameleon Cloud** that are mounted to both training and inference nodes.

- Storage is independent of containers, so data is preserved even if compute instances are recreated. It will be used to store:
  - Model checkpoints
  - Final trained models
  - Training logs and evaluation metrics

---

Offline Data Management

- Offline data includes:
  - Pre-registered student facial images
  - Synthetic images for model robustness evaluation
  - Public datasets for pretraining

- Copy of the above-mentioned data will be stored on persistent disk

---

Data Pipelines

- Extract:
  - Images from the dataset are stored in persistent storage

- Transform:
  - Preprocessing: resize, normalize, encode to embeddings
  - Label verification or correction
  - Format conversion to model input structure

- Load:
  - Transformed data is moved and loaded for the model to train on

- False positives and false negatives from user feedback from the inference node will be moved to persistent storage to be used in re-training

---

Online Data

- A simulated stream mimics real-time images captured at exam entry gates


### Continuous X

Infrastructure-as-Code:
- All infrastructure is provisioned using Terraform for resources, IPs and volumes on Chameleon Cloud.
- Ansible Playbooks are used to configure and deploy:
  - Triton Inference Server
  - MLflow
  - Prometheus
  - API services (FastAPI)
 
---

Cloud-Native Architecture:
- Immutable Infrastructure:
  - Infrastructure changes are made via pull requests to Git, then re-provisioned.
- Microservices:
  - The system is split into containers: API Server, Inference, Monitoring, Training, Testing, Storage
  - Each container communicates via APIs
- Containerization:
  - All services are Dockerized and deployed with Kubernetes.
  - Model training and inference environments are decoupled and reproducible.

---

CI/CD and Continuous Training Pipeline:
- ArgoCD power our CI/CD and retraining pipelines:
  - Triggered on schedule (ideally per semester to include new students).

---

Staged Deployment:
- Services will be promoted from one environment to another using AgroCD.
- Staging:
  - Load-tested with student entries.
- Canary:
  - Small percentage of live exam verifications pass through new model.
  - Metrics monitored via Prometheus.
- Production:
  - Model promoted if no degradation is observed during canary phase.





