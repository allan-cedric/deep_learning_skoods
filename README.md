## Trabalho final da disciplina de Robótica Móvel (CI1020) - Ciência da Computação - UFPR

*   Controle de um carro autônomo no ambiente simulado da Skoods.

### Ambiente da simulação

Acesse o sistema da [Skoods](https://github.com/skoods-org/Welcome) para baixar o ambiente da simulação.

Além disso vale a pena dar uma conferida no projeto open source, [Microsoft AirSim](https://github.com/microsoft/AirSim).

### Observações

*   Esse README.md serve para dar uma visão macro sobre os arquivos deste projeto, detalhes adicionais estão comentados nos scripts presentes nesse repositório.
*   É recomendado fortemente a consulta dos links acima para entender a estrutura da simulação.

*   O sistema aplicado aqui foi desenvolvido e testado no sistema operacional **Windows 10 64-bits**.
*   **Instale em um ambiente Anaconda exatamente o arquivo de pacotes e dependências: package-list.txt**, utilize o comando `conda create -n <myenv> --file package-list.txt` para criar um novo ambiente com todos esses pacotes. Tudo que precisa para rodar esse projeto está no arquivo `package-list.txt`.
*   **Ative o ambiente com o comando** `conda activate <myenv>`.

### Dataset

O formato padrão do *dataset* possui duas estruturas: um diretório com as imagens `(Ex.: images/)` e um arquivo de log com os *steering commands* do carro no formato *tsv - Tab Separated Values* `(Ex.: airsim_rec.txt)` .

Para ajudar na hora de gravar os dados, `airsim_dataset.py` é um script que possui uma classe que grava os dados da simulação, e gera um dataset novo em folha no padrão visto acima.

O script `__main__.py` já possui integração da classe de `airsim_dataset.py` . 

### Treinamento da rede neural

`train_nvidia_model.py` : Script de treinamento de uma rede neural convolucional baseada no modelo proposto pela NVIDIA no paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).

Esse script irá criar uma estrutura de diretórios com os melhores modelos neurais `(Ex.: nvidia_model/models/*.h5)`, sendo *.h5* a extensão padrão de um modelo neural.

### Teste da rede neural

`test_nvidia_model.py` : Script de teste da rede neural convolucional em um ambiente de simulação integrado com o Microsoft AirSim. Serve principalmente para debugar a rede neural.

### Rodar a simulação

`__main__.py` : Script principal para rodar o carro autônomo tanto com o controle PID simples fornecido pela própria Skoods, quanto com a rede neural convolucional. **Lembre-se de sempre rodar o ambiente da simulação antes de rodar esse script**.
