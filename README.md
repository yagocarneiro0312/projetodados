# Projeto Dados — Detector clássico com PyQt6 e OpenCV

Ferramenta para demonstrar a estatística de dois dados com leitura por webcam
ou fotografia, histórico de jogadas e percentuais das médias de cada par.
O detector identifica faces claras e conta seus pontos escuros, seguindo o
pipeline de `Detector_Dados_OpenCV_Passo_a_Passo.pdf`. Um simulador externo
permite comparar amostras de 200 ou 300 jogadas com a distribuição triangular
teórica.

Ao executar, o programa procura e abre automaticamente a **EMEET SmartCam S600**
com **1280 × 720 a 60 fps**. Se esse formato não estiver disponível, usa a qualidade automática.
O **threshold adaptativo gaussiano já inicia ativo nas faces**. Os pontos usam
**Otsu calculado dentro de cada face**, com escala normalizada, inclusive nas fotos.
O perfil inicial é o **Perfil EMEET**, ajustado para dados pequenos sobre fundo
estampado: Gaussian Blur 3×3, fechamento desativado, abertura 5×5, retangularidade
mínima de 0,80 e circularidade mínima dos pontos de 0,70. Isso evita unir a face
de um dado aos desenhos do mousepad e rejeita regiões pouco retangulares do fundo,
inclusive com faces giradas. O detector conta os dados e os pontos encontrados,
sem impor uma quantidade ou valores esperados.
O perfil preenche buracos internos de regiões compactas antes da abertura e
amplia o cinza de faces pequenas para pelo menos 128 px antes de segmentar os
pontos. Isso evita fragmentar faces com muitos pontos e reduz a sensibilidade
da contagem ao ângulo e à grade de pixels. As mesmas regras atendem às faces 1–6.
O Perfil EMEET também ativa **Separar dados encostados**: uma região com formato
de duas faces justapostas é dividida pelo eixo do retângulo girado. Cada parte
precisa passar pelos filtros de face e conter de 1 a 6 pontos; a divisão é
rejeitada se cortar um ponto. Isso permite contar pares lado a lado, na vertical
ou na diagonal, e atualizar o gráfico com os valores individuais. A opção fica
em **Filtros → Faces**. Sobreposição que esconda uma face ou seus pontos ainda
pode impedir uma leitura completa.
O **Perfil C920** continua disponível em Filtros. Suas medidas e valores
estão em [CALIBRACAO_C920.md](CALIBRACAO_C920.md).
Se ela estiver ausente, informa o problema e permite abrir uma foto ou selecionar
outra webcam manualmente. A captura usa Qt Multimedia; o processamento usa OpenCV.

## Executar

Com Python 3.10 ou superior, abra um terminal na pasta do projeto e execute:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Se necessário, use `py` no primeiro comando. Não é preciso ativar o ambiente.
As pastas `.venv` e `.tools` são locais e ignoradas pelo Git.
Depois da instalação inicial, basta executar `.\.venv\Scripts\python.exe main.py`.

## Interface

- **Área vermelha:** vídeo ou foto, mantendo a proporção, com marcações verdes
  nos dados aceitos, amarelas nos candidatos rejeitados e azuis nos pontos válidos.
- **Área azul/violeta:** câmera, resolução, zoom digital de 1× a 4×, brilho de
  −100 a +100, restauração dos ajustes, abertura de foto e retorno à EMEET.
- **Área verde:** imagens das etapas, parâmetros, análise das faces e resumo
  com dimensões, tempo, candidatos, quantidade de dados, valores e soma total.
- **Área amarela:** abas **Histórico** e **Agora**. O histórico mostra as últimas
  50 jogadas, os percentuais de cada média do par e as médias mais frequentes.
  A aba Agora mantém o gráfico em tempo real com Dado 1, Dado 2, Total e Divisão.
  O total soma os dois primeiros dados (da esquerda para a direita);
  a divisão é esse total dividido por 2, exibindo decimais quando necessário.
  O gráfico acompanha cada análise da webcam, a cada 100 ms, e também fotografias.
  Sem dois dados reconhecidos, Total e Divisão ficam sem valor. Pausar a análise
  ou fechar a câmera limpa o gráfico.

Em janelas menores, o painel lateral possui rolagem vertical para manter todos
os controles acessíveis.

### Histórico das jogadas

Na câmera, uma jogada é registrada quando **exatamente dois dados** mantêm seus
valores e posições estáveis por **1 segundo**. O resultado permanece bloqueado
para não contar novamente a cada frame. Retire ambos os dados da imagem por pelo
menos **0,8 segundo**, ou clique em **Nova jogada**, para liberar o próximo registro.
Isso permite registrar novamente os mesmos valores. Perder apenas um dos dados
ou interromper brevemente a detecção não libera uma nova jogada.

Cada foto aberta com exatamente dois dados reconhecidos registra uma jogada;
reprocessar a foto ou ajustar seus filtros não duplica o registro já confirmado.
**Nova jogada** permite registrar a mesma foto novamente.

As barras agrupam as jogadas pela **média dos dois dados**, calculada por
`(Dado 1 + Dado 2) ÷ 2`, com resultados de **1 a 6**, em passos de **0,5**.
Cada jogada conta uma vez. O percentual é a quantidade de jogadas com aquela média
dividida pelo total de jogadas registradas. Por exemplo, os pares **2 e 5** e
**3 e 4** contam para a mesma média **3,5**; se forem duas de quatro jogadas,
a barra de 3,5 mostra **50%**. O comprimento das barras usa a escala de 0 a 100%.
A média geral considera as médias de todas as jogadas da amostra. Empates na
maior frequência mostram todas as médias empatadas. A tabela mostra primeiro as jogadas mais recentes,
com os valores individuais e a média de cada par. Ao ultrapassar 50 jogadas, a
mais antiga deixa de participar das estatísticas.

**Desfazer** remove a última leitura e recalcula as estatísticas; **Limpar** esvazia
a amostra. Ambos mantêm o bloqueio da jogada atual para não registrá-la outra vez
automaticamente. O histórico é mantido durante a sessão, inclusive ao pausar o
processamento e trocar de fonte, e é descartado ao fechar o aplicativo.

### Simulação para demonstrações e artigos

A ferramenta externa **Simulador Estatístico de Dados** abre separadamente do
detector, com **200 ou 300 jogadas simuladas**. Dê dois cliques em
**Abrir simulador.bat**, ou execute:

```powershell
.\.venv\Scripts\python.exe simulador.py
```

Instale as dependências da ferramenta com
`.\.venv\Scripts\python.exe -m pip install -r requirements-simulation.txt`.
O detector não possui botão de simulação nem depende do Matplotlib.

Cada jogada sorteia dois inteiros uniformes de
1 a 6, de forma independente, e calcula a média do par. Todas as jogadas entram
no gráfico da simulação; a janela móvel de 50 registros vale apenas para o
histórico da câmera. As amostras simuladas não alteram as leituras da câmera.

As barras mostram os percentuais simulados, com as oscilações naturais de uma
amostra finita. A linha tracejada mostra as probabilidades exatas das onze médias:
`(1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1) ÷ 36`. Essa distribuição discreta tem formato
triangular, com maior probabilidade na média **3,5**. A amostra é gerada uma vez,
sem ajustar contagens ou repetir sorteios para impor uma forma específica.

**Gerar amostra** produz novos resultados. A semente exibida permite reproduzir
os mesmos pares: informe-a no campo opcional e mantenha a quantidade de jogadas.
**Salvar gráfico** exporta PNG em 300 dpi; **Exportar CSV** salva todos os pares,
suas médias, a semente e a origem `simulacao`. A tela e o gráfico identificam os
resultados como simulados, para uso como demonstração computacional no artigo.

Para gerar os exemplos de 200 e 300 jogadas fora da interface:

```powershell
.\.venv\Scripts\python.exe tools/generate_simulation_examples.py
```

Os arquivos são gravados em `docs/simulacao`. Use `--seed NUMERO` para reproduzir
uma execução. Os dois tamanhos usam a mesma sequência: a amostra de 300 contém
as primeiras 200 jogadas e mais 100 sorteios.

**Abrir imagem** aceita JPEG, PNG, BMP e TIFF, inclusive caminhos com acentos.
Uma imagem válida libera a webcam e usa o mesmo pipeline. Cancelar ou tentar
abrir um arquivo inválido conserva a fonte e os resultados atuais.
**Voltar à EMEET** conserva a foto até receber o primeiro frame válido da câmera;
se a câmera estiver ausente, ocupada ou não responder, a foto permanece disponível.

Zoom e brilho são aplicados por software tanto à visualização quanto à análise;
o original da foto permanece intacto. A resolução é um controle exclusivo da
webcam. Na captura, mudar a resolução reinicia a câmera e mantém os parâmetros.
O aplicativo libera a câmera ao fechar e não grava vídeo.

**Processar** pausa ou retoma a análise, limpando as marcações e contagens ao
pausar. A webcam continua exibindo vídeo. O processamento consulta o frame mais
recente a cada 100 ms, sem acumular uma fila; fotos são processadas ao abrir e
quando os ajustes mudam, sem repetir o cálculo continuamente.

As oito miniaturas são: colorida redimensionada, cinza, Gaussian Blur, threshold,
fechamento, abertura, candidatos e dados, resultado final. A faixa possui rolagem
horizontal. O resultado final marca somente os dados aceitos. As máscaras usam
vizinho mais próximo nas miniaturas para preservar preto e branco.

Passe o mouse sobre uma miniatura **ou seu título** para ver a etapa ampliada
na área principal. Retirar o mouse restaura a câmera ou a foto com suas anotações.
Na webcam, a etapa ampliada continua atualizando ao vivo. A inspeção não altera
a fonte nem reaplica zoom ou brilho à imagem já processada.

### Análise das faces

Abre uma captura estável dos resultados enquanto a webcam continua funcionando.
Selecione um candidato na lista para ver:

1. Recorte original da face.
2. Máscara preenchida do contorno.
3. Pontos escuros após threshold invertido, limitados à face.
4. Máscara dos pontos após abertura.
5. Contornos dos pontos válidos sobre o recorte.

A janela mostra área, área relativa, proporção, extent, solidez e retangularidade da face. A tabela mede
cada contorno escuro: área, perímetro, área relativa à face, proporção,
circularidade e motivos de rejeição. Um candidato com zero ou mais de seis
pontos válidos é rejeitado. Candidatos são as regiões que já passaram pelos
filtros geométricos das faces.

## Pipeline e padrões do PDF

| Etapa | Padrão |
| --- | --- |
| Redimensionamento | Largura 1000 px; interpolação linear; mesma escala nos dois eixos |
| Cinza | `cv2.COLOR_BGR2GRAY` |
| Suavização | Gaussian Blur 5×5, sigma automático |
| Segmentação das faces | `THRESH_BINARY`: intensidades acima de 190 são brancas |
| Fechamento | Kernel elíptico 5×5; duas iterações |
| Abertura das faces | Mesmo kernel; uma iteração |
| Contornos | `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE` |
| Área da face / imagem | `0.0005 < área relativa < 0.03` (0,05% a 3%) |
| Proporção da face | `0.55 < largura/altura < 1.60` |
| Extent da face | `área / (largura × altura) > 0.35` |
| Segmentação dos pontos | Cinza anterior à morfologia; threshold invertido 130; máscara da face |
| Limpeza dos pontos | Abertura elíptica 3×3, uma iteração |
| Área do ponto / face | `0.004 < área relativa < 0.10` (0,4% a 10%) |
| Circularidade do ponto | `4π × área / perímetro² > 0.55` |
| Proporção do ponto | `0.60 < largura/altura < 1.50` |
| Aceitação do dado | Entre 1 e 6 pontos válidos |

Os limites geométricos são **estritos**: igualdade com um limite rejeita a região.
O threshold invertido considera escuras intensidades menores ou iguais a 130.
O aplicativo inicia com threshold adaptativo gaussiano nas faces e Otsu nos pontos.
A tabela acima descreve os valores de referência do PDF, disponíveis ao escolher
o modo fixo ou restaurar os padrões do PDF. Restaurar o PDF também desativa o
preenchimento antes da abertura e a normalização de escala dos pontos.
Os dados aceitos recebem IDs da esquerda para a direita, recalculados em cada
imagem: `Dado 1: 4`, `Dado 2: 6`, `Soma total: 10`. Não há rastreamento temporal.

### Parâmetros ajustáveis

**Filtros** abre três abas: **Preparação**, **Faces** e **Pontos**. Todos os
valores da tabela que são parâmetros numéricos podem ser ajustados, incluindo
largura, thresholds, kernels, iterações e limites geométricos. Zero iterações
desativa aquela operação morfológica. A regra de aceitação de 1 a 6 pontos e
os tipos de contorno permanecem fixos.

**Perfil C920** repõe a calibração medida na cena real: área das faces entre
0,10% e 3%, proporção entre 0,70 e 1,40, extent acima de 0,45 e solidez acima
de 0,85; área dos pontos entre 0,8% e 7,5% da face e circularidade acima de 0,65.
Clique em Aplicar para usar esses valores. A solidez é área / área da envoltória
convexa e rejeita contornos irregulares. Zero desativa esse filtro adicional.
**Preencher máscara convexa** está ativo nesse perfil e recupera pontos excluídos
por pequenas reentrâncias na máscara. Os limites geométricos continuam sendo
calculados sobre o contorno original. O modo do PDF deixa essa opção desativada.

**Aplicar** valida e substitui conjuntamente os parâmetros; **Cancelar** conserva
a configuração; **Restaurar padrões do PDF** repõe os valores no diálogo, que
passam a valer após Aplicar. Kernels precisam ser positivos e ímpares; thresholds
ficam entre 0 e 255; os limites devem ser finitos e o mínimo menor que o máximo.
As legendas e os resultados são atualizados após aplicar. Ajustes valem durante
a sessão e não são gravados em disco.

### Threshold adaptativo

Em **Filtros → Preparação**, escolha **Modo de threshold das faces**. Em
**Filtros → Pontos**, escolha o modo para as manchas escuras. Cada etapa permite
**Fixo (PDF)**, **Adaptativo gaussiano**, **Adaptativo pela média** ou **Otsu (automático)**,
de forma independente. Otsu desabilita os controles de threshold fixo, bloco e C;
nos pontos, usa somente os pixels cobertos pela máscara da face para escolher o limite.
No modo adaptativo, o valor fixo é desabilitado; ajuste o **Bloco
adaptativo** e a **Constante C**, depois clique em Aplicar.

O threshold adaptativo calcula o limite local como a média (simples ou ponderada
por Gaussian) menos C. Pode ajudar quando a iluminação varia entre regiões;
o bloco e C precisam ser calibrados para a imagem. Comece com bloco **101** e
**C = −10** para faces claras, e bloco **31** e **C = 7** para pontos escuros.
O bloco deve ser ímpar e pelo menos 3. Um C negativo nas faces exige que os
pixels sejam mais claros que a média local; nos pontos, o threshold é invertido.
Os modos adaptativos preservam a morfologia e os filtros geométricos existentes.
Restaurar padrões do PDF retorna ambos os modos ao threshold fixo.

**Preencher face antes da abertura**, em Preparação, completa buracos internos
somente em regiões compactas. Desenhos extensos do fundo não são preenchidos.
**Normalizar escala dos pontos**, em Pontos, amplia o cinza com interpolação linear
antes de aplicar o threshold e a abertura. Nesse modo, o kernel dos pontos usa a
escala ampliada, e uma aproximação do contorno reduz os degraus do raster na
medição da circularidade. Áreas, raios, perímetros e coordenadas exibidos são convertidos
de volta à escala da imagem de análise. Nenhum valor esperado entra na decisão.

### Cenas sintéticas de rotação

O gerador determinístico produz faces com cantos arredondados, rotações,
perspectiva, variação de posição, contraste, ruído e desfoque. A validação compara
o perfil EMEET anterior ao robusto em 408 cenas positivas de faces 1–6. Os testes
também incluem faces vazias, sete pontos e marcas alongadas para conferir rejeições.

- [Dado 6 em 24 rotações, de 0° a 345°](docs/synthetic_dice/six_rotations.png).
- [Comparação das faces de 1 a 6](docs/synthetic_dice/all_values.png).
- [Imagens sintéticas individuais sem anotações](docs/synthetic_dice/six/).
- [Resultados por cena e parâmetros usados](docs/synthetic_dice/validation.json).

Para regenerar as imagens e os resultados:

```powershell
.\.venv\Scripts\python.exe tools/generate_synthetic_validation.py
```

Essas cenas verificam os casos simulados e complementam as referências reais
C920 e EMEET dos testes. Não substituem a validação da iluminação e foco da câmera.

Referência: [threshold adaptativo no OpenCV](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).

O detector foi elaborado para dados claros com pontos escuros sobre mesa mais
escura. Reflexos, sombras, dados encostados e perspectivas difíceis podem alterar
a contagem; use as máscaras e as métricas para calibrar a iluminação e os filtros.

## Organização e interfaces

- `main.py`: interface, fontes de imagem, câmera padrão, controles e renderização.
- `pipeline_parameters.py`: parâmetros validados e padrões do PDF.
- `vision_pipeline.py`: pré-processamento, via `preprocess_frame(img, parameters=None)`.
- `dice_detection.py`: faces e pontos, via `detect_candidates(result, filters=None, pip_filters=None)`.
- `dice_history.py`: últimas 50 jogadas, estatísticas e confirmação de resultados estáveis.
- `dice_chart.py`: gráficos atuais, frequências e tabela de histórico.
- `dice_simulation.py`: sorteios reproduzíveis, CSV e gráfico da distribuição das médias.
- `simulador.py` e `Abrir simulador.bat`: iniciadores da ferramenta externa de simulação.
- `dice_simulation_dialog.py`: janela independente de 200 ou 300 jogadas simuladas.
- `candidate_settings.py`: diálogo de parâmetros.
- `face_analysis.py`: captura e análise dos candidatos.
- `image_source.py`: leitura de fotos com suporte a caminhos Unicode.
- `thresholding.py`: segmentação fixa, adaptativa e Otsu das faces e dos pontos.
- `stage_preview.py`: miniaturas e inspeção ao passar o mouse.
- `qt_image_utils.py`: conversões com cópia dos pixels entre Qt e OpenCV.
- `tests/`: testes de processamento, parâmetros, interface e fotografias de referência.

As funções do pipeline recebem imagens BGR `uint8`. Chamadas sem configuração
usam os padrões do PDF. Os resultados incluem imagens intermediárias, candidatos,
faces isoladas, máscaras, medidas de cada ponto, IDs, `dice_values`, `total_sum`,
imagem anotada de candidatos e `final_bgr` somente com dados aceitos. Anotações
não entram no processamento. Os campos anteriores dos resultados foram mantidos.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes automatizados não exigem webcam física. Cobrem o pré-processamento,
limites estritos, pontos de 1 a 6, rejeição de 0 e 7, máscaras, circularidade,
parâmetros, Unicode, pausa, ajustes, diálogo de configuração, captura estável da
análise, layout e troca de fontes. Câmeras simuladas verificam erros, timeout,
desconexão e conservação da foto. As referências incorporadas reproduzem os
resultados **[4, 6], soma 10**, e **[6], soma 6**.
Também verificam segmentação adaptativa em iluminação desigual e pontos acima do
threshold fixo, controles dos modos, ampliação por mouse, atualização da etapa
com novos frames e retorno à fonte ao sair ou interromper o processamento.

Verificado com a **HD Pro Webcam C920 real** em 05/10/2026:

- Inicialização de `main.py` sem cliques em 1920 × 1080, com pipeline ativo.
- Zoom, brilho, pausa, retomada e restauração dos ajustes.
- Troca de resolução para 640 × 480 durante a transmissão.
- Abertura da foto, liberação da câmera e retorno à C920 com resolução conservada.
- Fechamento e liberação do dispositivo.
- Ausência simulada da C920 mantendo a câmera integrada fechada.

A desconexão física ainda deve ser conferida manualmente; o fluxo correspondente
é coberto com dispositivo simulado. Se a webcam não abrir, confira sua conexão,
o uso por outro aplicativo e a permissão de câmera para aplicativos da área de
trabalho nas configurações do Windows.

O perfil adaptativo calibrado foi conferido na **C920 real** com as faces 4, 5 e 6:
**60 frames salvos e 20 novos frames ao vivo** produziram exatamente três dados,
com soma 15. Os testes também incluem quatro amostras reais, a remoção de um
falso dado e a recuperação de um ponto cortado pela máscara da face.
