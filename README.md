# Synthetic Image Detection

Detecting AI-generated images has become a crucial task. This project builds upon the work of [1] with the goal of evaluating possible improvements through the **Smash and Reconstruction (S&R)** technique. S&R suppresses semantic information in images and focuses on inter-pixel correlations. For further details, refer to [2].

### Project Scope
This project aims to compare:
- The **Smash and Reconstruction** technique against common approaches.
- Performance across **CNNs, ResNet18, and SuSy** models.

### Key Idea
The underlying hypothesis is straightforward: if **CNNs and ResNet** models benefit from **S&R**, then **SuSy** models might also gain performance improvements.

### Computational Considerations
A key challenge of this project is computational resources. Thus, results should not be considered absolute. Instead, this project encourages further exploration using a larger dataset, as well as experimenting with different architectures or alternative approaches.

### Reproducing the Experiment
To replicate the experiment, download the SID_Notebook.ipynb file and the Utils_Funct folder. Since the project was developed on Google Colab, simply upload the notebook and the mentioned folder into the Colab workspace to get started.

---

### References

[1] Bernabeu-Perez, Pablo and Lopez-Cuena, Enrique and Garcia-Gasulla, Dario. "Present and Future Generalization of Synthetic Image Detectors." *arXiv preprint arXiv:2409.14128*, 2024.

[2] Zhong, Nan and Xu, Yiran and Qian, Zhenxing and Zhang, Xinpeng. "Rich and poor texture contrast: A simple yet effective approach for AI-generated image detection." *arXiv preprint arXiv:2311.12397*, 2023.
