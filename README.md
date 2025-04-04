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

#Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->
![System Diagram](images/system_diagram.png)


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
| `compute_skylake` | 2 for entire project                           | 1 in model serving and 1 for model evaluation and load testing           |
| `gpu_v100`     | 4 gpus/ 4 slots of 5 hours                        |       Required for training the ResNet-50 for the large database        |
| Floating IPs    | 2 running throghout the project                |    1 for model serving api, 1 for monitoring while training, testing and serving  |
| Persistent Volume  - 'block storage'   |                1 volume - 10GB                                  |       Needed to store model checkpoints, logs, retrained models, and OpenVINO outputs        |
| Object Storage 'CHI@TACC Swift' |   1 volume - 50GB     |  Storing the static dataset

## Detailed design plan
---

### Model training and infrastructure platforms

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

### Model serving and monitoring platforms

#### Serving from an API endpoint:
During serving, a FastAPI endpoint will exist through which we will get a request from the user (input) and return the result, i.e. verified or not (output)

#### Model Requirements:
We will have 2 instances of our model, one with high throughput and concurrency for online exams multiple students will check in at the same time for an exam, and one with concurrency 1 for offline exams as students will check in one at a time at one exam hall  
Size: 100MB  
**Throughput:**  
In person exams - 100-200 req/second (low concurrency)  
Online exams - 800 req/second (high concurrency and dynamic batching)  

**Single sample latency:** predicting a latency of 10-20ms  
**Concurrency requirement:**  
- Concurrency of 1 for offline exams  
- Concurrency of 8 for online exams  

#### Model optimizations to satisfy requirements:
Since we want to explore and use different optimization techniques and execution providers, we will be converting our model to Onnx format.

**Graph optimizations:** We will be using graph optimizations like fusing - combining common subsequent operations and constant folding - precomputing operations where inputs are constant.  
Since these optimizations don't reduce performance, we will implement them.

**Quantizations:** Since we require high accuracy and are trying to prevent false negatives we will experiment with conservative quantization but will most likely not use any quantization techniques.

#### System optimizations to satisfy requirements:
**Backend:** We will use the triton inference server as the backend as it is optimized for high throughput, it has inbuilt support for concurrency and monitoring with prometheus and support for different execution providers.  
**OpenVino:** We will use the OpenVino EP which is optimized for intel CPUs for high throughput on CPU hardware.  

---

#### Multiple options for serving:
We will have 2 options for serving:  
- 1 model will be served with a high concurrency with dynamic batching for evaluation in online exams where multiple students at the same exam will be scanning at the same time.  
- 1 model will be served with a concurrency of 1 since it will be needed for a throughput of only 150-250 req/sec

---

#### Offline evaluation of model:
We will have a suit of offline tests that will run immediately after the model training and unit testing. These tests will be triggered automatically via an internal api between the training and testing microservices.  
**Standard and domain specific tests:** This will be from the VGGFace2 dataset subset with appropriate weightage of people types similar to that in production (e.g. ethnic distributions specific to NYU)  
**Population and slices:** We will create mini sets of different types of data, like different ethnicities, skin colours, facial hair, lighting conditions  
**Known Failure cases:** We will consider bad fail cases to test against like blurry photos, bad lighting, similar faces of different people.

The results of this testing and evaluation will be automatically saved in MLFlow which will be accessed from a through a floating IP. If a certain threshold of each test type is passed, the model will be automatically saved to the model registry via MLFlow.

---

#### Load test in staging:
Once the model accuracy in different conditions has been validated, we will now stress test our model to see if it matches our throughput and latency requirements in the same environment as our production/serving environment, i.e. Triton server with OpenVino on the compute_skylake node.

**Load testing will be done in 2 ways:**  
- Testing for the single concurrency model which is served for offline exams, required throughput - > 150 req/sec and latency < 100ms  
- Testing for multiple concurrency of 8 with dynamic batching for online exams, required throughput > 750 req/sec and latency < 500ms  
(Since people will not be lining up in online exams behind the camera a higher latency per user is not a huge deal)

---

#### Online evaluation in canary
During the canary rollout, the new model is deployed to a limited number of exam entry gates to assess its performance in real-world conditions before full-scale deployment. This phase simulates actual student behavior, including scenarios such as poor lighting, delays in entry, and multiple verification attempts caused by initial misdetections. The purpose is to observe how the model performs under imperfect and variable conditions, ensuring robustness and reliability before promotion to production.

---

#### Close the loop:
There will be feedback from the user via a separate API, which will automatically register incorrect evaluations (False Negatives) when a proctor has to verify a correct identity which is not verified by the model or when the user needs multiple tries to pass the verification.  
In addition to this, when new users are added to the database, cameras will pick up facial data during class for the scheduled retraining. This data will finish the feedback loop for model retraining, which will happen automatically when the model performance in monitoring is deteriorated to a certain extent or a large number of new users are added.

---

#### Business specific metrics:
- Improvement in security and reduced fraud/cheating:  
There should be a reduction in imposter’s caught by proctors in the exam hall or in online exams.

- Efficiency:  
In place of taking manual attendance we should see a much faster system with automated attendance. If we record two groups, one with manual attendance and one with the ML system, we should see a decrease in total time for conducting the examination.

---

### Data pipeline

#### Persistent Storage

- We provision block storage volumes on **Chameleon Cloud** that are mounted to both training and inference nodes.
- Storage is independent of containers, so data is preserved even if compute instances are recreated. It will be used to store:
  - Model checkpoints
  - Final trained models
  - Training logs and evaluation metrics

---

#### Offline Data Management

- Offline data includes:
  - Pre-registered student facial images
  - Synthetic images for model robustness evaluation
  - Public datasets for pretraining
- Copy of the above-mentioned data will be stored on persistent disk

---

#### Data Pipelines

- **Extract**:
  - Images from the dataset are stored in persistent storage
- **Transform**:
  - Preprocessing: resize, normalize, encode to embeddings
  - Label verification or correction
  - Format conversion to model input structure
- **Load**:
  - Transformed data is moved and loaded for the model to train on
- False positives and false negatives from user feedback from the inference node will be moved to persistent storage to be used in re-training

---

#### Online Data

- A simulated stream mimics real-time images captured at exam entry gates

---

### Continuous X

#### Infrastructure-as-Code

- All infrastructure is provisioned using Terraform for resources, IPs and volumes on Chameleon Cloud.
- Ansible Playbooks are used to configure and deploy:
  - Triton Inference Server
  - MLflow
  - Prometheus
  - API services (FastAPI)

---

#### Cloud-Native Architecture

- **Immutable Infrastructure**:
  - Infrastructure changes are made via pull requests to Git, then re-provisioned.
- **Microservices**:
  - The system is split into containers: API Server, Inference, Monitoring, Training, Testing, Storage
  - Each container communicates via APIs
- **Containerization**:
  - All services are Dockerized and deployed with Kubernetes.
  - Model training and inference environments are decoupled and reproducible.

---

#### CI/CD and Continuous Training Pipeline

- ArgoCD power our CI/CD and retraining pipelines:
  - Triggered on schedule (ideally per semester to include new students).

---

#### Staged Deployment

- Services will be promoted from one environment to another using AgroCD.
- **Staging**:
  - Load-tested with student entries.
- **Canary**:
  - Small percentage of live exam verifications pass through new model.
  - Metrics monitored via Prometheus.
- **Production**:
  - Model promoted if no degradation is observed during canary phase.



