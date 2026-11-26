# 🐾 Classificação Automática da Fauna Silvestre Brasileira com Deep Learning

Sistema de Visão Computacional baseado em Redes Neurais Convolucionais (**MobileNetV2**) e *Transfer Learning* para identificação automática de cinco espécies da fauna silvestre brasileira: **Anta** (*Tapirus terrestris*), **Capivara** (*Hydrochoerus hydrochaeris*), **Lobo-Guará** (*Chrysocyon brachyurus*), **Onça-Pintada** (*Panthera onca*) e **Tamanduá-Bandeira** (*Myrmecophaga tridactyla*).

---

## 📋 Sobre o Projeto

O monitoramento da fauna por meio de armadilhas fotográficas (*camera traps*) gera um volume massivo de imagens, tornando a triagem manual lenta e sujeita a falhas humanas. Este projeto automatiza a classificação dessas espécies utilizando uma arquitetura leve e eficiente, ideal para aplicações de ecologia computacional e sistemas embarcado em campo.

### Tecnologias Utilizadas
* **Linguagem:** Python 3
* **Deep Learning:** TensorFlow / Keras (Arquitetura `MobileNetV2` pré-treinada na ImageNet)
* **Processamento e Métricas:** NumPy, Pillow, Scikit-Learn, Matplotlib, Seaborn

---

## 🗂️ Estrutura do Repositório

```text
├── dados_animais/          # Dataset organizado por classes (treino e validação)
├── assets/                 # Gráficos, matriz de confusão e capturas de tela
├── treinar.py              # Script de pré-processamento, Data Augmentation e treinamento
├── interface.py            # Interface para inferência e teste em novas imagens
├── modelo_animais_brasileiros.keras  # Modelo treinado salvo
├── classes.json            # Mapeamento dos índices das espécies
└── requirements.txt        # Dependências do projeto