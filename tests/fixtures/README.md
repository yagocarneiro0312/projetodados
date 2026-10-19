# Imagens de referência

- `pdf_two_dice.png`: painel fotográfico esquerdo da figura da página 1 de
  `Detector_Dados_OpenCV_Passo_a_Passo.pdf`, fornecido pelo usuário. Recorte sem
  título ou margens; resultados esperados: 4 e 6, soma 10.
- `photo_six.png`: fotografia `IM001.jpg` disponível em Downloads, correspondente
  ao exemplo de um dado do documento. Orientação aplicada pelo OpenCV e largura
  reduzida para 1000 px com interpolação linear; resultado esperado: 6.

Os arquivos foram incorporados para que os testes não dependam de Downloads,
do PDF original ou de uma webcam. Essas duas fotografias não são frames capturados da câmera.

## Cena C920 com três dados

`c920_456/frame_00.png`, `frame_01.png`, `frame_13.png` e `frame_33.png` são frames reais da
HD Pro Webcam C920, capturados para a calibração solicitada pelo usuário.
O vídeo de 1920×1080 foi redimensionado proporcionalmente para 1000×562 com
interpolação linear, sem anotações. As faces visíveis são 6, 4 e 5, nessa ordem
da esquerda para a direita; a soma correta é 15.

As duas primeiras amostras incluem falsos dados aceitos pelos filtros anteriores;
a terceira já era contada corretamente. O perfil calibrado deve reconhecer
exatamente as três faces em todas as amostras.
`frame_33.png` pertence a uma segunda sequência e reproduz a perda do quarto
ponto em uma reentrância da máscara original. A máscara convexa recupera o ponto.

## Cena EMEET com dois dados

`emeet_54/` contém sete frames reais sem anotações da EMEET SmartCam S600
em 1280×720 a 60 fps, capturados em 08/10/2026 para reproduzir o problema
relatado pelo usuário. As faces são 5 e 4 sobre um mousepad estampado, com soma 9.
O perfil C920 unia a face 4 aos desenhos claros durante o fechamento em várias
amostras. Os frames 09, 24 e 26 reproduzem também falsos positivos no fundo
quando a retangularidade está desativada. Os testes conferem ambas as faces
e a ausência de dados adicionais.

## Cena EMEET com dados encostados

`emeet_touching_13/` contém quatro frames reais sem anotações, capturados em
08/10/2026 com a EMEET em 1280×720 a 60 fps. As faces 1 e 3 encostam e formam
um único contorno alongado, que era rejeitado antes da separação. A leitura
esperada é dois dados, valores 1 e 3, soma 4 e divisão por 2 igual a 2.
