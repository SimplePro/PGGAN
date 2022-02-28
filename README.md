PGGAN
-------------------------------
### result
- 약간의 모드붕괴 현상이 보여지고 있지만, 비교적 적은 양의 데이터셋으로 학습한 것을 고려하면 납득되는 성능이다.    
     
![image](https://user-images.githubusercontent.com/66504341/155912625-dd70e567-3bae-437e-afdc-342201690735.png)
![image](https://user-images.githubusercontent.com/66504341/155912665-e3b9cfe3-c362-4105-bb4b-93ba1ff0d066.png)

### train log
- Google Colab Pro 환경에서 학습하였다.
- 약 4~5일간 학습하였고, 각 resolution에서 만족하는 성능이 나올때까지 여러번 학습하여 학습시간이 더 오래 걸렸다.
- epoch마다 모델 파라미터를 백업해두어 학습이 끝난 후에 원하는 파라미터를 사용해볼까 싶었지만, PGGAN 구조상 fade in 레이어에서 사용되는 alpha가 1이 되어야 해서 학습 중간의 모델 파라미터 상태를 최종 모델의 파라미터로 사용할 수가 없었다. (alpha값은 학습중에 선형적으로 0 to 1로 증가한다, alpha가 1이 되는 때가 마지막 epoch 때이다.)

[![PGGAN Train Log](https://img.youtube.com/vi/pvSaE_BVKJM/0.jpg)](https://www.youtube.com/watch?v=pvSaE_BVKJM)


### dataset
- https://www.kaggle.com/andrewmvd/animal-faces
- 위의 데이터셋에서 cat과 dog만 사용하였다.


### paper
- https://arxiv.org/pdf/1710.10196.pdf

### blog
- https://simplepro.tistory.com/44
- https://simplepro.tistory.com/45, https://simplepro.tistory.com/46, https://simplepro.tistory.com/47
