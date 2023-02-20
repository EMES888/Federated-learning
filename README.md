# Federated-learning
In this repository we try to implement Federated learning algorithms on text data(Shakespear Dataset).
Primal and dual manner in optimization is critical point of solving optimization problem and when this two methods meet **Federated learning** it can help to solve problems iterativly.
The matter in federated learning is that we don't have centralized dataset and for privacy or other reasons we have distributed data between clients(in here 10) and try to learn even our data is not in one centralize place.
Our task here is **Next character prediction** and we used **FedAVG**, **Fedprox**, **Liadmm** and  **FedAdmm** algorithms for solving our problem.
## Preprocessing
At first we preprocessed our data to be suitable for giving that data to model and learn and after that we did some visualization of unigram and bigram of our dataset
## Implementation
At first we use LSTM model as our baseline for our centralized training and after that implement decomposition methods for our decomposed dataset. Spliting data between clients happend in two ways **IID** and **non-IID** in IID amount of data for each client is equal and in non-iid distributions between clients are different and in all methods clients have same LSTM network with same common parameters
## Compare results
By comparing results we understand most methods gets better results in IID distribution except Liadmm which trained in fewer steps than other algorithm and FedAVG and FedProx and FedAdmm results are close even fedprox in some loss parts are not as good as those other two. and finally centralize gives best loss curve and we know that centralize is suitable way but if we can't have that other decompositon methods have approprate results.

![all_acc_IID](https://user-images.githubusercontent.com/58938036/220176655-35bd89d3-1aae-43fb-a86c-3ebd91605a24.png)
![all_acc_non-iid](https://user-images.githubusercontent.com/58938036/220176750-cbad2f93-2e58-41a0-8bfd-e14ce180eba7.png)

![all_loss_with_centralize_IID](https://user-images.githubusercontent.com/58938036/220176229-19fc6153-0cdf-40a0-b4f8-2f1d373bfe87.png)
![all_loss_with_centralized_non-iid](https://user-images.githubusercontent.com/58938036/220176567-58835162-8fd3-4360-b57e-c4899ce56d66.png)
