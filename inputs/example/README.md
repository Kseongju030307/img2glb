여기에 캐릭터 이미지를 넣으세요 (파일명 고정):

  front.png   (필수)  정면
  back.png    (선택)  후면
  left.png    (선택)  왼쪽 측면
  right.png   (선택)  오른쪽 측면

- png / jpg / webp 가능
- 투명 배경(RGBA) 이면 그대로 사용, 배경이 있으면 자동으로 rembg 배경 제거
- 4장 모두 같은 크기 / 같은 스케일 / 같은 캐릭터 포즈(T포즈 또는 A포즈 권장) 로 맞추면 결과가 좋음
- 캐릭터가 이미지 중앙에 오도록, 여백 10~15% 정도 두기

캐릭터 여러 개면 inputs/ 아래 폴더를 하나씩 더 만들면 됨:
  inputs/charA/front.png ...
  inputs/charB/front.png ...
그 뒤  bash batch.sh  로 GPU 4장에 나눠 한 번에 처리.
