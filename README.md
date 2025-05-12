# MLOps Project


## Facial Recognition system for student authentication at NYU

# 
Value proposition: <br>
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

![System Diagram](MLOps.png)


## Summary of outside materials

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| MS1MV2       |  	Cleaned and refined MS-Celeb-1M dataset (~5.8M images, 85K IDs)                  |       Free for use           |
| VGGFace2     |    Collected via Google Images; 3.3M images, 9K+ identities                |         Research only          |
| iResNet-50   |    Deep residual network trained on large face datasets               |          Open-source         |
| MTCNN        |    Trained on various public datasets, by MIT                |         Open source         |

Common dataset link: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

## Summary of infrastructure requirements

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1-medium` | 2 for entire project                           | 1 in model serving and 1 for model evaluation and load testing           |
| `gpu_v100`     | 4 gpus/ 4 slots of 5 hours                        |       Required for training the ResNet-50 for the large database        |
| Floating IPs    | 2 running throghout the project                |    1 for model serving api, 1 for monitoring while training, testing and serving  |
| Persistent Volume  - 'block storage'   |                1 volume - 10GB                                  |       Needed to store model checkpoints, logs, retrained models, and OpenVINO outputs        |
| Object Storage 'CHI@TACC Swift' |   1 volume - 50GB     |  Storing the static dataset

## Detailed design plan
---

## Unit 3 and 4: Model training and infrastructure

- We will employ mini-batch Stochastic Gradient Descent (SGD) for training, with the batch size determined by GPU memory. Gradient accumulation will be explored to simulate larger batch sizes if memory becomes a constraint.
- To optimize memory usage, we will consider mixed-precision training (FP16 gradients with an FP32 optimizer) and monitor the memory footprint of the chosen optimizer.
- To accelerate training, we will implement distributed data parallelism across multiple GPUs available on Chameleon. This involves:
  - Replicating the model across each GPU.
  - Dividing the training data into slices, with each GPU processing a different slice.
  - Synchronizing gradients across all GPUs using an all-reduce operation to ensure consistent model updates.
  - Utilizing PyTorch Distributed to manage data distribution and gradient communication.
- Training will be conducted on the Chameleon cloud platform using bare-metal GPU nodes, in our case 4x v100 GPUs. A Jupyter notebook server on a reserved instance will be used for interactive development.
- We will containerize our environment with Docker, including all dependencies (Python, PyTorch, dataset libraries) to ensure reproducibility. MLFlow will track experiments (parameters, metrics, checkpoints, code versions) via a tracking server accessible through a specified port.
- For distributed training, we will explore Ray and potentially Ray Tune for hyperparameter tuning, following lab instructions for setup and job submission.
- The face recognition datasets (MS1MV2 and VGGFace2) will be properly organized and mounted for access within the container. GPU and system performance will be monitored via Jupyter or the Chameleon dashboard, and resources will be properly released after training to avoid extra costs.

---

## Unit 6: Model serving:

### -Serving from an API endpoint:
We have wrapped our model in a fastapi backend application which runs on a seperate node_mode_serve_project-14 so that its performance is uninterrupted by trainnig and testing. It has a simple '/compare' endpoint which taks 2 image files as input, creates their embeddings using the model and then checks if they are the ame using a threshold for cosine similarity for the embeddings. 
You can find the application code and the docker-compose file to create a container and run the app and the other serving infrastructure at [model_serve](https://github.com/MoulikShah/MLOps_Project/tree/main/model_serve)

### -Identify requirements:
Since we are running an offline service that will only handle concurrent users at entrances to exam halls, the throughput of the system is not very important. 
However we would like a short latency so that our system does not cause delays as each student is entering the classroom 1 by 1. These are the requirements
Throughput: > 10 req/sec
Latency: < 500ms

### -Model optimizations to satisfy requirements:
Since we want to explore and use different optimization techniques and execution providers, we will be converting our model to Onnx format.

**Graph optimizations:** We will be using graph optimizations like fusing - combining common subsequent operations and constant folding - precomputing operations where inputs are constant.  
Since these optimizations don't reduce performance, we will implement them.

**Quantizations:** Since we require high accuracy and are trying to prevent false negatives we will experiment with conservative quantization but will most likely not use any quantization techniques.

### -System optimizations to satisfy requirements:
**Backend:** We will use a simple fastapi server as the backend as it is simple, light and matches our throughput/latency requirements  

---

## Unit 7: Evaluation and monitoring

### -Offline evaluation of model: 
We have the following tests for offline monitoring: 
1) Standard tests: This consists of tests with postiive pairs (2 images of same person) and negative pairs (1 anchor image of the person with a randomly selected image).
   Domain specific tests: pThese results are more significant as they show ius how the odel will behave with real world inputs. Here we used a pretrained model, deepface to obtain age gender and ethnicity for each identity. Positive pairs for people under the eage of 30 were taken.
   Negative paird were chosen for identites with the same ethnicity and gender, since it is morelikely for fraud cases.
2) Population slices: We have created subsets for the following population slices: Indians and middle eastern, Asian, White and Black. We ave also created subsets based on gender.
3) Test on known failure modes: Here we have handpciked samples from the domain specific set (same ethicity and gender) that look particularly similar, we have also picked some cases with bad lighting and blur, cases we believe the model may struggle.
4) Template based tests:

The results of this testing and evaluation will be automatically saved in MLFlow which will be accessed from a through a floating IP. If a certain threshold of each test type is passed, the model will be automatically saved to the model registry via MLFlow.

### -Load test in staging: 
After our model passes all the offline tests,, it will be moved to the staging area, here we will pas in a large subset for load testing and display the results: 
  Throughput 
  Latency

### -Online evaluation in canary:
Here we will conduct an online evaluation, which is when we use data similar to real users, (ages < 35, ethnicity split: 1/3rd Indian, 1/3rd Asian, 1/3rd white and black). 
Our data will be sent to our base model as well as our newly trained model, and we will compare our results, new model will only be moved forward if it fairs better than the base model.

### -Close the loop:
Here we assume to get feedback in 2 ways. Positive feedback will be automatically sent back as a v small subset of correctly predicted cases.
For negative feedback there can be 1 of two cases - If the person does not get correctly recognized and the professor or staff has to manually verify, If a person manages to cheat the model and happens to get caught. we will specifically label data of similar looking people usnig label studio. rest of the negative feedback data wil be automated since we have ground truth labels.

### -Business specific metrics:
- Improvement in security and reduced fraud/cheating:  
If we happen to record the number of instances that a person has been caught cheating per year or semester, we can check these results pre and post our ML implementation. 
if there is a decrease, we know that thee has been an increase in academic integrity.

- Efficiency:  
In place of taking manual attendance we should see a much faster system with automated attendance. If we record two groups, one with manual attendance and one with the ML system, we should see a decrease in total time for conducting the examination.

---

## Unit 8: Data pipeline

### Persistent Storage Infrastructure

Our face recognition system utilizes *Chameleon Cloud's persistent storage* to manage large-scale datasets and artifacts, decoupling data from compute resources.

#### Object Storage (CHI@TACC Swift)

- *Size:* 20GB allocated for immutable datasets  
- *Container Name:* object-persist-project-14  
- *Access Method:* Mounted as *read-only* via rclone  
- *Contents:*
  - MS1MV2 dataset (~5.8M images, 85K identities)
  - Test verification sets categorized by:
    - Ethnicity
    - Quality factors

#### Block Storage (KVM@TACC)

- *Size:* 18GB allocated for mutable data  
- *Volume Name:* block-persist-group14  
- *Access Method:* Mounted as ext4 at /mnt/block  
- *Contents:*
  - Model checkpoints
  - Generated face embeddings
  - Evaluation metrics and logs
  - MLflow experiment tracking artifacts


### Offline Data: Training Data Organization

Face recognition datasets follow a strict organizational structure, enabling efficient training and evaluation across multiple benchmarks and applications.

### Data Pipeline Workflow

#### 1. 🧪 Extract

- *MS1MV2:* Subset of MS-Celeb-1M (cleaned)
- *VGGFace2:* Scraped from Google Images
- *NYU Test Data:* Synthetically generated with desired distributions

#### 2. 🧼 Transform

- Face detection using *MTCNN*  
- Image normalization: 112x112 px, RGB  
- Sample filtering based on:
  - Clarity
  - Alignment
  - Size  
- Data augmentation:
  - Brightness & contrast shifts
  - Rotations

#### 3. 🧵 Load

- Transformed data uploaded to *object storage*
- Training pipeline:
  - Loads batches with efficient caching
  - Stores generated *embeddings* on block storage


#### Data Quality and Validation

- Pre-processing checks for:
  - Minimum face clarity & alignment
  - Balanced demographics (ethnicity, gender, age)
  - Duplicate removal

#### Production Data Simulation

Simulated student check-in scenarios include:

- Dynamic lighting (e.g., classroom lighting)
- Occlusions: Glasses, masks, hair
- Motion blur (low to high)
- Random batch arrivals


#### Data Flow for Retraining

- *False negatives* captured via feedback API  
- Problem samples saved to block storage  
- Regular retraining incorporates flagged cases  
- New student data handled by separate onboarding pipeline

#### 🛠 Scripts and Tools

- mount_object_store.sh: Mounts object storage with caching optimizations  
- setup_rclone.sh: Sets up rclone credentials  
- Docker containers mount storage volumes for training pipeline access

### Online Data

- A simulated stream mimics real-time images captured at exam entry gates

---

## Unit 3: Continuous X

### Infrastructure-as-Code

- All infrastructure is provisioned using Terraform for resources, IPs and volumes on Chameleon Cloud.
- Ansible Playbooks are used to configure and deploy:
  - Triton Inference Server
  - MLflow
  - Prometheus
  - API services (FastAPI)

---

### Cloud-Native Architecture

- **Immutable Infrastructure**:
  - Infrastructure changes are made via pull requests to Git, then re-provisioned.
- **Microservices**:
  - The system is split into containers: API Server, Inference, Monitoring, Training, Testing, Storage
  - Each container communicates via APIs
- **Containerization**:
  - All services are Dockerized and deployed with Kubernetes.
  - Model training and inference environments are decoupled and reproducible.

---

### CI/CD and Continuous Training Pipeline

- ArgoCD power our CI/CD and retraining pipelines:
  - Triggered on schedule (ideally per semester to include new students).

---

### Staged Deployment

- Services will be promoted from one environment to another using AgroCD.
- **Staging**:
  - Load-tested with student entries.
- **Canary**:
  - Small percentage of live exam verifications pass through new model.
  - Metrics monitored via Prometheus.
- **Production**:
  - Model promoted if no degradation is observed during canary phase.


## Project Structure Overview

### `Terraform/`

This directory contains the infrastructure-as-code configuration using **Terraform** to provision resources on Chameleon Cloud.
Different subdirectories have the code for provisioning resources on different sites.

- `main.tf`: Defines the virtual machines (VMs), their images, flavors (e.g., `m1.medium`), networks, and ports.
- `variables.tf`: Declares input variables like `skylake_id` for lease binding.
- `data.tf`: Fetches existing OpenStack resources like networks and security groups.
- `outputs.tf`: Displays IPs and other relevant outputs after provisioning.
- `terraform.tfvars`: Stores actual values for input variables used during apply.

### `Ansible/`

This directory contains automation scripts for configuring the provisioned infrastructure using **Ansible**.

- `inventory.yml`: Defines the hosts (e.g., `node1`, `node2`) with their corresponding IP addresses and SSH users. Ansible uses this inventory to connect and manage each node.
- `playbook.yml`: The main playbook that includes tasks for setting up the software environment on the nodes—such as installing Docker, Kubernetes components, and other dependencies.

### `Setup Files/`

The `Setup Files/` directory contains utility and scripts that prepare each node for further configuration and deployment. It is also used to clone this repository to the local Jupyter interface and install dependencies for terraform, ansible and kubernetes.



