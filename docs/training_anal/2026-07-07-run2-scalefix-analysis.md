# SDUM run2 (`sdum_scalefix`) 학습 로그 분석 (2026-07-07)

run1에서 드러난 P0(손실이 안 떨어짐)를 **어댑터 per-sample 스케일 정규화**(커밋 `c56351d`)로 고친 뒤 처음 돌린 학습. iter 2500에서 다른 실험을 위해 수동 인터럽트(Ctrl-C)했다.

> **한 줄 결론:** 스케일 정규화는 **성공했다** — 손실이 run1의 1.0 고정에서 ~0.1대로 내려왔다. 다만 이건 **"안 망가짐"이지 "잘한다"가 아직 아니고**(SSIM 바닥 효과), 속도(≈9h/epoch)는 **그대로**다(`--num-workers`는 효과 없었음).
python train.py --model sdum --num-cascades 4 --channels 16 32   -b 1 -l 2e-4 -e 1 -r 10 --num-workers 4   -n sdum_scalefix -t /root/Data/train/ -v /r
oot/Data/val/
Current cuda device:  0
BasicUNet features: (12, 24, 48, 96, 192, 12).
BasicUNet features: (12, 24, 48, 96, 192, 12).
BasicUNet features: (12, 24, 48, 96, 192, 12).
BasicUNet features: (12, 24, 48, 96, 192, 12).
Epoch # 0 ............... sdum_scalefix ...............
Epoch = [  0/  1] Iter = [   0/4651] Loss = 0.2035 Time = 7.2613s
Epoch = [  0/  1] Iter = [  10/4651] Loss = 0.1383 Time = 68.2230s
Epoch = [  0/  1] Iter = [  20/4651] Loss = 0.1606 Time = 69.7204s
Epoch = [  0/  1] Iter = [  30/4651] Loss = 0.1848 Time = 69.3172s
Epoch = [  0/  1] Iter = [  40/4651] Loss = 0.1314 Time = 68.7829s
Epoch = [  0/  1] Iter = [  50/4651] Loss = 0.1351 Time = 69.9161s
Epoch = [  0/  1] Iter = [  60/4651] Loss = 0.2786 Time = 71.8699s
Epoch = [  0/  1] Iter = [  70/4651] Loss = 0.09558 Time = 71.5523s
Epoch = [  0/  1] Iter = [  80/4651] Loss = 0.1477 Time = 69.5398s
Epoch = [  0/  1] Iter = [  90/4651] Loss = 0.188 Time = 69.3061s
Epoch = [  0/  1] Iter = [ 100/4651] Loss = 0.07991 Time = 69.9659s
Epoch = [  0/  1] Iter = [ 110/4651] Loss = 0.1372 Time = 70.0191s
Epoch = [  0/  1] Iter = [ 120/4651] Loss = 0.1528 Time = 68.1082s
Epoch = [  0/  1] Iter = [ 130/4651] Loss = 0.1105 Time = 69.2241s
Epoch = [  0/  1] Iter = [ 140/4651] Loss = 0.09853 Time = 70.0053s
Epoch = [  0/  1] Iter = [ 150/4651] Loss = 0.2303 Time = 69.9627s
Epoch = [  0/  1] Iter = [ 160/4651] Loss = 0.05114 Time = 69.7760s
Epoch = [  0/  1] Iter = [ 170/4651] Loss = 0.1039 Time = 69.7846s
Epoch = [  0/  1] Iter = [ 180/4651] Loss = 0.1729 Time = 67.6922s
Epoch = [  0/  1] Iter = [ 190/4651] Loss = 0.07981 Time = 69.5160s
Epoch = [  0/  1] Iter = [ 200/4651] Loss = 0.1855 Time = 71.5925s
Epoch = [  0/  1] Iter = [ 210/4651] Loss = 0.09893 Time = 70.2402s
Epoch = [  0/  1] Iter = [ 220/4651] Loss = 0.1943 Time = 69.3955s
Epoch = [  0/  1] Iter = [ 230/4651] Loss = 0.08218 Time = 69.4589s
Epoch = [  0/  1] Iter = [ 240/4651] Loss = 0.1957 Time = 70.0256s
Epoch = [  0/  1] Iter = [ 250/4651] Loss = 0.2063 Time = 69.9769s
Epoch = [  0/  1] Iter = [ 260/4651] Loss = 0.1256 Time = 71.2261s
Epoch = [  0/  1] Iter = [ 270/4651] Loss = 0.2813 Time = 68.9445s
Epoch = [  0/  1] Iter = [ 280/4651] Loss = 0.1404 Time = 69.0688s
Epoch = [  0/  1] Iter = [ 290/4651] Loss = 0.2021 Time = 68.6845s
Epoch = [  0/  1] Iter = [ 300/4651] Loss = 0.2316 Time = 68.5175s
Epoch = [  0/  1] Iter = [ 310/4651] Loss = 0.06402 Time = 71.4467s
Epoch = [  0/  1] Iter = [ 320/4651] Loss = 0.1308 Time = 68.6216s
Epoch = [  0/  1] Iter = [ 330/4651] Loss = 0.1912 Time = 69.6259s
Epoch = [  0/  1] Iter = [ 340/4651] Loss = 0.1937 Time = 68.4405s
Epoch = [  0/  1] Iter = [ 350/4651] Loss = 0.2065 Time = 70.1175s
Epoch = [  0/  1] Iter = [ 360/4651] Loss = 0.1674 Time = 69.4063s
Epoch = [  0/  1] Iter = [ 370/4651] Loss = 0.1844 Time = 69.4990s
Epoch = [  0/  1] Iter = [ 380/4651] Loss = 0.2014 Time = 70.0465s
Epoch = [  0/  1] Iter = [ 390/4651] Loss = 0.06385 Time = 69.6133s
Epoch = [  0/  1] Iter = [ 400/4651] Loss = 0.2206 Time = 68.6728s
Epoch = [  0/  1] Iter = [ 410/4651] Loss = 0.1075 Time = 69.5892s
Epoch = [  0/  1] Iter = [ 420/4651] Loss = 0.1035 Time = 70.0624s
Epoch = [  0/  1] Iter = [ 430/4651] Loss = 0.1966 Time = 70.2093s
Epoch = [  0/  1] Iter = [ 440/4651] Loss = 0.2215 Time = 71.0408s
Epoch = [  0/  1] Iter = [ 450/4651] Loss = 0.1215 Time = 69.6423s
Epoch = [  0/  1] Iter = [ 460/4651] Loss = 0.07346 Time = 70.2200s
Epoch = [  0/  1] Iter = [ 470/4651] Loss = 0.08134 Time = 69.4087s
Epoch = [  0/  1] Iter = [ 480/4651] Loss = 0.1263 Time = 69.6125s
Epoch = [  0/  1] Iter = [ 490/4651] Loss = 0.2177 Time = 69.9593s
Epoch = [  0/  1] Iter = [ 500/4651] Loss = 0.08228 Time = 70.0251s
Epoch = [  0/  1] Iter = [ 510/4651] Loss = 0.1121 Time = 69.9782s
Epoch = [  0/  1] Iter = [ 520/4651] Loss = 0.09178 Time = 69.8099s
Epoch = [  0/  1] Iter = [ 530/4651] Loss = 0.04145 Time = 71.0940s
Epoch = [  0/  1] Iter = [ 540/4651] Loss = 0.1162 Time = 69.1408s
Epoch = [  0/  1] Iter = [ 550/4651] Loss = 0.1388 Time = 69.3209s
Epoch = [  0/  1] Iter = [ 560/4651] Loss = 0.08816 Time = 69.5544s
Epoch = [  0/  1] Iter = [ 570/4651] Loss = 0.1553 Time = 68.4298s
Epoch = [  0/  1] Iter = [ 580/4651] Loss = 0.1836 Time = 69.6194s
Epoch = [  0/  1] Iter = [ 590/4651] Loss = 0.1992 Time = 65.7866s
Epoch = [  0/  1] Iter = [ 600/4651] Loss = 0.1208 Time = 69.6853s
Epoch = [  0/  1] Iter = [ 610/4651] Loss = 0.0542 Time = 69.2267s
Epoch = [  0/  1] Iter = [ 620/4651] Loss = 0.107 Time = 69.7906s
Epoch = [  0/  1] Iter = [ 630/4651] Loss = 0.1435 Time = 69.1616s
Epoch = [  0/  1] Iter = [ 640/4651] Loss = 0.164 Time = 69.3789s
Epoch = [  0/  1] Iter = [ 650/4651] Loss = 0.0879 Time = 69.6513s
Epoch = [  0/  1] Iter = [ 660/4651] Loss = 0.231 Time = 70.0863s
Epoch = [  0/  1] Iter = [ 670/4651] Loss = 0.08761 Time = 66.5703s
Epoch = [  0/  1] Iter = [ 680/4651] Loss = 0.1041 Time = 69.6365s
Epoch = [  0/  1] Iter = [ 690/4651] Loss = 0.1011 Time = 69.4895s
Epoch = [  0/  1] Iter = [ 700/4651] Loss = 0.2863 Time = 69.6372s
Epoch = [  0/  1] Iter = [ 710/4651] Loss = 0.08385 Time = 69.2291s
Epoch = [  0/  1] Iter = [ 720/4651] Loss = 0.2239 Time = 69.8410s
Epoch = [  0/  1] Iter = [ 730/4651] Loss = 0.07152 Time = 69.8840s
Epoch = [  0/  1] Iter = [ 740/4651] Loss = 0.09298 Time = 69.2496s
Epoch = [  0/  1] Iter = [ 750/4651] Loss = 0.06898 Time = 67.1666s
Epoch = [  0/  1] Iter = [ 760/4651] Loss = 0.07575 Time = 68.5931s
Epoch = [  0/  1] Iter = [ 770/4651] Loss = 0.1884 Time = 69.2012s
Epoch = [  0/  1] Iter = [ 780/4651] Loss = 0.07837 Time = 69.7162s
Epoch = [  0/  1] Iter = [ 790/4651] Loss = 0.1323 Time = 69.8737s
Epoch = [  0/  1] Iter = [ 800/4651] Loss = 0.08823 Time = 68.7807s
Epoch = [  0/  1] Iter = [ 810/4651] Loss = 0.158 Time = 70.9400s
Epoch = [  0/  1] Iter = [ 820/4651] Loss = 0.07263 Time = 67.5362s
Epoch = [  0/  1] Iter = [ 830/4651] Loss = 0.07086 Time = 71.2273s
Epoch = [  0/  1] Iter = [ 840/4651] Loss = 0.1033 Time = 70.3039s
Epoch = [  0/  1] Iter = [ 850/4651] Loss = 0.09547 Time = 68.3621s
Epoch = [  0/  1] Iter = [ 860/4651] Loss = 0.0631 Time = 69.0648s
Epoch = [  0/  1] Iter = [ 870/4651] Loss = 0.1284 Time = 69.1204s
Epoch = [  0/  1] Iter = [ 880/4651] Loss = 0.1559 Time = 69.7910s
Epoch = [  0/  1] Iter = [ 890/4651] Loss = 0.01712 Time = 69.9538s
Epoch = [  0/  1] Iter = [ 900/4651] Loss = 0.0823 Time = 70.9243s
Epoch = [  0/  1] Iter = [ 910/4651] Loss = 0.08425 Time = 69.3372s
Epoch = [  0/  1] Iter = [ 920/4651] Loss = 0.06217 Time = 69.7916s
Epoch = [  0/  1] Iter = [ 930/4651] Loss = 0.09904 Time = 69.8377s
Epoch = [  0/  1] Iter = [ 940/4651] Loss = 0.08961 Time = 69.1916s
Epoch = [  0/  1] Iter = [ 950/4651] Loss = 0.06629 Time = 69.4150s
Epoch = [  0/  1] Iter = [ 960/4651] Loss = 0.1194 Time = 69.8414s
Epoch = [  0/  1] Iter = [ 970/4651] Loss = 0.09783 Time = 69.3237s
Epoch = [  0/  1] Iter = [ 980/4651] Loss = 0.04004 Time = 69.9514s
Epoch = [  0/  1] Iter = [ 990/4651] Loss = 0.06825 Time = 68.9433s
Epoch = [  0/  1] Iter = [1000/4651] Loss = 0.09297 Time = 69.5173s
Epoch = [  0/  1] Iter = [1010/4651] Loss = 0.1191 Time = 70.0609s
Epoch = [  0/  1] Iter = [1020/4651] Loss = 0.08365 Time = 68.9346s
Epoch = [  0/  1] Iter = [1030/4651] Loss = 0.09304 Time = 68.1765s
Epoch = [  0/  1] Iter = [1040/4651] Loss = 0.0805 Time = 68.2401s
Epoch = [  0/  1] Iter = [1050/4651] Loss = 0.1401 Time = 70.3230s
Epoch = [  0/  1] Iter = [1060/4651] Loss = 0.1019 Time = 69.6005s
Epoch = [  0/  1] Iter = [1070/4651] Loss = 0.05499 Time = 69.4022s
Epoch = [  0/  1] Iter = [1080/4651] Loss = 0.128 Time = 70.0572s
Epoch = [  0/  1] Iter = [1090/4651] Loss = 0.1085 Time = 68.6182s
Epoch = [  0/  1] Iter = [1100/4651] Loss = 0.06591 Time = 69.9096s
Epoch = [  0/  1] Iter = [1110/4651] Loss = 0.08947 Time = 69.8405s
Epoch = [  0/  1] Iter = [1120/4651] Loss = 0.1271 Time = 71.1511s
Epoch = [  0/  1] Iter = [1130/4651] Loss = 0.09027 Time = 68.3160s
Epoch = [  0/  1] Iter = [1140/4651] Loss = 0.1537 Time = 69.9078s
Epoch = [  0/  1] Iter = [1150/4651] Loss = 0.09613 Time = 69.6683s
Epoch = [  0/  1] Iter = [1160/4651] Loss = 0.08848 Time = 69.0440s
Epoch = [  0/  1] Iter = [1170/4651] Loss = 0.07564 Time = 70.2387s
Epoch = [  0/  1] Iter = [1180/4651] Loss = 0.1648 Time = 69.8836s
Epoch = [  0/  1] Iter = [1190/4651] Loss = 0.2572 Time = 69.2131s
Epoch = [  0/  1] Iter = [1200/4651] Loss = 0.1836 Time = 71.3072s
Epoch = [  0/  1] Iter = [1210/4651] Loss = 0.1793 Time = 69.5021s
Epoch = [  0/  1] Iter = [1220/4651] Loss = 0.08874 Time = 70.0892s
Epoch = [  0/  1] Iter = [1230/4651] Loss = 0.1293 Time = 71.3491s
Epoch = [  0/  1] Iter = [1240/4651] Loss = 0.1218 Time = 70.2001s
Epoch = [  0/  1] Iter = [1250/4651] Loss = 0.05965 Time = 69.9541s
Epoch = [  0/  1] Iter = [1260/4651] Loss = 0.09975 Time = 70.4286s
Epoch = [  0/  1] Iter = [1270/4651] Loss = 0.2138 Time = 70.0520s
Epoch = [  0/  1] Iter = [1280/4651] Loss = 0.2641 Time = 69.1074s
Epoch = [  0/  1] Iter = [1290/4651] Loss = 0.1802 Time = 69.1264s
Epoch = [  0/  1] Iter = [1300/4651] Loss = 0.05807 Time = 69.4994s
Epoch = [  0/  1] Iter = [1310/4651] Loss = 0.07087 Time = 69.3760s
Epoch = [  0/  1] Iter = [1320/4651] Loss = 0.08618 Time = 69.7815s
Epoch = [  0/  1] Iter = [1330/4651] Loss = 0.1708 Time = 70.0368s
Epoch = [  0/  1] Iter = [1340/4651] Loss = 0.05002 Time = 69.9603s
Epoch = [  0/  1] Iter = [1350/4651] Loss = 0.08123 Time = 71.0480s
Epoch = [  0/  1] Iter = [1360/4651] Loss = 0.2254 Time = 67.3481s
Epoch = [  0/  1] Iter = [1370/4651] Loss = 0.1021 Time = 70.2835s
Epoch = [  0/  1] Iter = [1380/4651] Loss = 0.07557 Time = 69.1872s
Epoch = [  0/  1] Iter = [1390/4651] Loss = 0.1237 Time = 69.5284s
Epoch = [  0/  1] Iter = [1400/4651] Loss = 0.09729 Time = 70.2272s
Epoch = [  0/  1] Iter = [1410/4651] Loss = 0.0975 Time = 69.5303s
Epoch = [  0/  1] Iter = [1420/4651] Loss = 0.2358 Time = 68.2582s
Epoch = [  0/  1] Iter = [1430/4651] Loss = 0.05692 Time = 69.4190s
Epoch = [  0/  1] Iter = [1440/4651] Loss = 0.1474 Time = 69.2449s
Epoch = [  0/  1] Iter = [1450/4651] Loss = 0.1824 Time = 69.8615s
Epoch = [  0/  1] Iter = [1460/4651] Loss = 0.1457 Time = 69.6199s
Epoch = [  0/  1] Iter = [1470/4651] Loss = 0.09759 Time = 69.6335s
Epoch = [  0/  1] Iter = [1480/4651] Loss = 0.1969 Time = 69.3277s
Epoch = [  0/  1] Iter = [1490/4651] Loss = 0.05255 Time = 69.8909s
Epoch = [  0/  1] Iter = [1500/4651] Loss = 0.1195 Time = 69.8107s
Epoch = [  0/  1] Iter = [1510/4651] Loss = 0.1067 Time = 68.9231s
Epoch = [  0/  1] Iter = [1520/4651] Loss = 0.08623 Time = 69.6387s
Epoch = [  0/  1] Iter = [1530/4651] Loss = 0.105 Time = 69.6911s
Epoch = [  0/  1] Iter = [1540/4651] Loss = 0.05746 Time = 70.4814s
Epoch = [  0/  1] Iter = [1550/4651] Loss = 0.04611 Time = 68.8254s
Epoch = [  0/  1] Iter = [1560/4651] Loss = 0.1882 Time = 68.5895s
Epoch = [  0/  1] Iter = [1570/4651] Loss = 0.05582 Time = 70.1801s
Epoch = [  0/  1] Iter = [1580/4651] Loss = 0.09508 Time = 70.3915s
Epoch = [  0/  1] Iter = [1590/4651] Loss = 0.1185 Time = 69.7439s
Epoch = [  0/  1] Iter = [1600/4651] Loss = 0.09137 Time = 69.9569s
Epoch = [  0/  1] Iter = [1610/4651] Loss = 0.1029 Time = 69.4662s
Epoch = [  0/  1] Iter = [1620/4651] Loss = 0.2684 Time = 68.7075s
Epoch = [  0/  1] Iter = [1630/4651] Loss = 0.06614 Time = 69.6403s
Epoch = [  0/  1] Iter = [1640/4651] Loss = 0.1078 Time = 69.2720s
Epoch = [  0/  1] Iter = [1650/4651] Loss = 0.07476 Time = 69.5853s
Epoch = [  0/  1] Iter = [1660/4651] Loss = 0.07583 Time = 69.9711s
Epoch = [  0/  1] Iter = [1670/4651] Loss = 0.07471 Time = 69.2194s
Epoch = [  0/  1] Iter = [1680/4651] Loss = 0.0615 Time = 71.0142s
Epoch = [  0/  1] Iter = [1690/4651] Loss = 0.1326 Time = 69.4438s
Epoch = [  0/  1] Iter = [1700/4651] Loss = 0.07802 Time = 68.0428s
Epoch = [  0/  1] Iter = [1710/4651] Loss = 0.05209 Time = 69.8387s
Epoch = [  0/  1] Iter = [1720/4651] Loss = 0.04578 Time = 69.3071s
Epoch = [  0/  1] Iter = [1730/4651] Loss = 0.1551 Time = 70.0454s
Epoch = [  0/  1] Iter = [1740/4651] Loss = 0.1392 Time = 69.8841s
Epoch = [  0/  1] Iter = [1750/4651] Loss = 0.1544 Time = 68.2656s
Epoch = [  0/  1] Iter = [1760/4651] Loss = 0.07179 Time = 69.6820s
Epoch = [  0/  1] Iter = [1770/4651] Loss = 0.0988 Time = 69.9370s
Epoch = [  0/  1] Iter = [1780/4651] Loss = 0.1235 Time = 70.1637s
Epoch = [  0/  1] Iter = [1790/4651] Loss = 0.1017 Time = 69.4565s
Epoch = [  0/  1] Iter = [1800/4651] Loss = 0.08622 Time = 70.3306s
Epoch = [  0/  1] Iter = [1810/4651] Loss = 0.06508 Time = 69.8866s
Epoch = [  0/  1] Iter = [1820/4651] Loss = 0.08894 Time = 69.2629s
Epoch = [  0/  1] Iter = [1830/4651] Loss = 0.1191 Time = 69.4568s
Epoch = [  0/  1] Iter = [1840/4651] Loss = 0.1928 Time = 69.8930s
Epoch = [  0/  1] Iter = [1850/4651] Loss = 0.1862 Time = 69.4660s
Epoch = [  0/  1] Iter = [1860/4651] Loss = 0.06362 Time = 69.2344s
Epoch = [  0/  1] Iter = [1870/4651] Loss = 0.2471 Time = 68.2857s
Epoch = [  0/  1] Iter = [1880/4651] Loss = 0.04013 Time = 69.8788s
Epoch = [  0/  1] Iter = [1890/4651] Loss = 0.08877 Time = 69.6385s
Epoch = [  0/  1] Iter = [1900/4651] Loss = 0.09424 Time = 70.2288s
Epoch = [  0/  1] Iter = [1910/4651] Loss = 0.06293 Time = 70.0511s
Epoch = [  0/  1] Iter = [1920/4651] Loss = 0.2529 Time = 69.4356s
Epoch = [  0/  1] Iter = [1930/4651] Loss = 0.1708 Time = 69.4635s
Epoch = [  0/  1] Iter = [1940/4651] Loss = 0.1104 Time = 69.5750s
Epoch = [  0/  1] Iter = [1950/4651] Loss = 0.1695 Time = 69.8271s
Epoch = [  0/  1] Iter = [1960/4651] Loss = 0.0646 Time = 69.9114s
Epoch = [  0/  1] Iter = [1970/4651] Loss = 0.1297 Time = 70.3328s
Epoch = [  0/  1] Iter = [1980/4651] Loss = 0.1072 Time = 69.6444s
Epoch = [  0/  1] Iter = [1990/4651] Loss = 0.08109 Time = 69.8545s
Epoch = [  0/  1] Iter = [2000/4651] Loss = 0.1172 Time = 66.8852s
Epoch = [  0/  1] Iter = [2010/4651] Loss = 0.08719 Time = 69.2656s
Epoch = [  0/  1] Iter = [2020/4651] Loss = 0.06544 Time = 69.1898s
Epoch = [  0/  1] Iter = [2030/4651] Loss = 0.1777 Time = 69.6364s
Epoch = [  0/  1] Iter = [2040/4651] Loss = 0.1753 Time = 69.6808s
Epoch = [  0/  1] Iter = [2050/4651] Loss = 0.09068 Time = 69.6323s
Epoch = [  0/  1] Iter = [2060/4651] Loss = 0.03619 Time = 68.3781s
Epoch = [  0/  1] Iter = [2070/4651] Loss = 0.1508 Time = 69.2916s
Epoch = [  0/  1] Iter = [2080/4651] Loss = 0.1533 Time = 69.1711s
Epoch = [  0/  1] Iter = [2090/4651] Loss = 0.1029 Time = 69.5032s
Epoch = [  0/  1] Iter = [2100/4651] Loss = 0.1577 Time = 69.6422s
Epoch = [  0/  1] Iter = [2110/4651] Loss = 0.1349 Time = 70.5560s
Epoch = [  0/  1] Iter = [2120/4651] Loss = 0.06363 Time = 68.2469s
Epoch = [  0/  1] Iter = [2130/4651] Loss = 0.2564 Time = 69.8527s
Epoch = [  0/  1] Iter = [2140/4651] Loss = 0.07193 Time = 69.7242s
Epoch = [  0/  1] Iter = [2150/4651] Loss = 0.07057 Time = 69.4133s
Epoch = [  0/  1] Iter = [2160/4651] Loss = 0.1054 Time = 71.1135s
Epoch = [  0/  1] Iter = [2170/4651] Loss = 0.09859 Time = 68.3738s
Epoch = [  0/  1] Iter = [2180/4651] Loss = 0.08536 Time = 69.6694s
Epoch = [  0/  1] Iter = [2190/4651] Loss = 0.08157 Time = 69.6835s
Epoch = [  0/  1] Iter = [2200/4651] Loss = 0.07215 Time = 69.8436s
Epoch = [  0/  1] Iter = [2210/4651] Loss = 0.07861 Time = 69.2302s
Epoch = [  0/  1] Iter = [2220/4651] Loss = 0.09933 Time = 69.6812s
Epoch = [  0/  1] Iter = [2230/4651] Loss = 0.1002 Time = 69.4908s
Epoch = [  0/  1] Iter = [2240/4651] Loss = 0.1001 Time = 69.5836s
Epoch = [  0/  1] Iter = [2250/4651] Loss = 0.1352 Time = 68.5358s
Epoch = [  0/  1] Iter = [2260/4651] Loss = 0.112 Time = 69.7944s
Epoch = [  0/  1] Iter = [2270/4651] Loss = 0.07041 Time = 69.7515s
Epoch = [  0/  1] Iter = [2280/4651] Loss = 0.0363 Time = 70.1473s
Epoch = [  0/  1] Iter = [2290/4651] Loss = 0.261 Time = 70.2406s
Epoch = [  0/  1] Iter = [2300/4651] Loss = 0.1519 Time = 69.5546s
Epoch = [  0/  1] Iter = [2310/4651] Loss = 0.1156 Time = 69.7517s
Epoch = [  0/  1] Iter = [2320/4651] Loss = 0.1269 Time = 68.4184s
Epoch = [  0/  1] Iter = [2330/4651] Loss = 0.2318 Time = 69.0049s
Epoch = [  0/  1] Iter = [2340/4651] Loss = 0.09214 Time = 70.1793s
Epoch = [  0/  1] Iter = [2350/4651] Loss = 0.1189 Time = 69.0380s
Epoch = [  0/  1] Iter = [2360/4651] Loss = 0.08799 Time = 69.6212s
Epoch = [  0/  1] Iter = [2370/4651] Loss = 0.1593 Time = 68.3787s
Epoch = [  0/  1] Iter = [2380/4651] Loss = 0.1896 Time = 69.1228s
Epoch = [  0/  1] Iter = [2390/4651] Loss = 0.09086 Time = 69.5307s
Epoch = [  0/  1] Iter = [2400/4651] Loss = 0.1762 Time = 70.4076s
Epoch = [  0/  1] Iter = [2410/4651] Loss = 0.1126 Time = 68.3850s
Epoch = [  0/  1] Iter = [2420/4651] Loss = 0.0475 Time = 69.6036s
Epoch = [  0/  1] Iter = [2430/4651] Loss = 0.09253 Time = 69.6108s
Epoch = [  0/  1] Iter = [2440/4651] Loss = 0.1277 Time = 68.5047s
Epoch = [  0/  1] Iter = [2450/4651] Loss = 0.04805 Time = 71.0125s
Epoch = [  0/  1] Iter = [2460/4651] Loss = 0.06237 Time = 69.6210s
Epoch = [  0/  1] Iter = [2470/4651] Loss = 0.2106 Time = 68.3048s
Epoch = [  0/  1] Iter = [2480/4651] Loss = 0.0909 Time = 71.2365s
Epoch = [  0/  1] Iter = [2490/4651] Loss = 0.07962 Time = 70.8690s
Epoch = [  0/  1] Iter = [2500/4651] Loss = 0.1902 Time = 69.6023s
^CTraceback (most recent call last):
  File "/root/FastMRI_SNU_2026_GMG/train.py", line 59, in <module>
    train(args)
  File "/root/FastMRI_SNU_2026_GMG/utils/learning/train_part.py", line 130, in train
    train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
  File "/root/FastMRI_SNU_2026_GMG/utils/learning/train_part.py", line 37, in train_epoch
    total_loss += loss.item()
KeyboardInterrupt
---

## 0. TL;DR

1. ✅ **P0 해결.** Loss가 run1의 **~1.0 고정(SSIM≈0)** → 이번엔 **0.03~0.28 (SSIM ~0.72~0.97, 중심 ~0.11)**. 어댑터 수정이 먹혔고 파이프라인이 이제 실제로 학습한다.
2. ⚠️ **과신 금지.** iter 0부터 SSIM 0.80이고 이후 노이즈 속 정체다. loss 0.1의 상당 부분은 실력이 아니라 **MRI SSIM의 원래 높은 바닥**일 수 있다 → 진짜 실력은 **val SSIM_bbox + zero-filled 대비**로 확인해야 한다.
3. ❌ **속도 그대로.** 여전히 **~6.9초/iter**, `--num-workers 4`가 **거의 효과 없음** → 병목은 데이터 로딩이 아니라 **GPU 연산**(compute-bound). 9시간/epoch 문제는 안 풀렸다.

---

## 1. 실행 조건

```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -l 2e-4 -e 1 -r 10 --num-workers 4 \
  -n sdum_scalefix -t /root/Data/train/ -v /root/Data/val/
```

**run1 대비 바뀐 점:**
- 어댑터에 **per-sample 스케일 정규화** 적용(입력 ÷s, 출력 ×s). ← 이번 검증의 핵심.
- `--num-workers 4`(run1은 0) — 데이터 로딩 병렬화.
- `-r 10`(run1은 50) — 로그 촘촘히.
- config·lr·epoch은 동일: **가장 작은 4×[16,32]**, AdamW lr=2e-4(스케줄러 없음), batch=1, 1 epoch.

**공통 조건:** train 4651 슬라이스/epoch, 손실 `SSIMLoss = 1 − SSIM(output, target, data_range=maximum)`. iter 2500(≈4.8시간)에서 수동 인터럽트 → **val 미실행**(아직 val SSIM 없음).

---

## 2. 손실 분석 — 핵심 성과 (해석 주의)

### 2.1 run1 vs run2

| | run1 | run2 (scalefix) |
|---|---|---|
| Loss 범위 | **~1.0 고정** | **0.03 ~ 0.28** |
| SSIM (=1−loss) | ≈ 0 | ≈ 0.72 ~ 0.97 |
| iter 0 (학습 전) | 1.0 | **0.2035 (SSIM 0.80)** |
| 추세 | 감소 없음 | 초반 하강 후 정체 |

run1은 **바닥(SSIM≈0)에서 안 움직였고**, run2는 **정상 출발선(SSIM≈0.8)에서 시작**한다. 질적으로 완전히 다른 상태다.

### 2.2 왜 iter 0부터 SSIM 0.80인가 — 스케일 정규화의 효과

`iter 0`은 **가중치 업데이트가 반영되기 전**의 첫 배치 손실이다. 그런데도 SSIM 0.80이 나온다는 건, 네트워크가 "학습해서"가 아니라 **출력 magnitude가 처음부터 타깃과 맞기 때문**이다.

- run1: 어댑터가 `ifft(masked_kspace)`를 **정규화 없이** 넣어 출력 절대 스케일이 타깃과 수십~수백 배 어긋남 → SSIM(스케일 민감)이 붕괴 → 0.
- run2: per-sample 스케일 `s`로 나눠 신경망에 ~단위 스케일로 주고 출력에 `s`를 다시 곱함 → 출력이 처음부터 타깃 물리 스케일 → **구조가 바로 잡힘**. (E2E-VarNet의 DC가 해주던 역할을 어댑터가 대신.)

즉 **iter 0의 SSIM 0.80 자체가 "스케일이 원인이었다"는 확정 증거**다.

### 2.3 학습 추세: 초반 하강 후 노이즈 정체

per-iter 손실은 **batch=1이라 매 iter가 서로 다른 슬라이스** → 난이도에 따라 크게 튄다(0.02~0.29). 그래서 구간 평균으로 봐야 한다.

**200-iter 구간 평균:**
| 구간(iter) | 평균 Loss | ≈SSIM |
|---|---|---|
| 0 – 200 | 0.146 | 0.85 |
| 500 – 700 | 0.128 | 0.87 |
| 900 – 1100 | 0.089 | 0.91 |
| 2300 – 2500 | 0.124 | 0.88 |

**250-iter 간격 샘플:** `0.20, 0.21, 0.082, 0.069, 0.093, 0.060, 0.12, 0.15, 0.12, 0.14, 0.19`

읽는 법:
- 첫 ~수백 iter에 **0.15 → 0.10대로 얕게 하강**(SSIM 0.85 → 0.90쯤).
- 그 뒤로는 **뚜렷한 지속 하강 없이 0.10~0.13 band에서 노이즈 정체**. 900–1100의 낮은 값(0.089)은 부분적으로 쉬운 슬라이스 운도 섞였을 것.
- **경고:** 학습 loss만으론 "계속 배우는 중인지 / 정체인지"를 못 가른다(노이즈 지배). **판단은 val로.**

### 2.4 ⚠️ loss 0.1을 곧이곧대로 믿지 말 것 — SSIM 바닥

MRI는 **zero-filled 복원만 해도 SSIM이 원래 0.7~0.85**로 높다(SSIM이 저주파 구조에 후하기 때문). 따라서 "loss 0.1 = SSIM 0.9"의 **상당 부분은 네트워크 실력이 아니라 SSIM의 높은 바닥**일 수 있다.

→ **P0(치명적 실패)는 확실히 풀렸지만, "복원 잘한다"는 아직 미확정.** 네트워크가 zero-filled 위에 **실제로 얼마를 더했는지**를 §4-A로 반드시 확인해야 한다. 특히 순위축인 **SSIM_bbox(병변 디테일)** 는 저주파 바닥의 덕을 못 보므로 여기서 진짜 실력이 드러난다.

---

## 3. 속도 분석 — `--num-workers`는 효과 없었다

### 3.1 숫자

`-r 10`이라 출력 `Time`은 **10-iter 블록** 소요시간:

| | run1 (workers=0) | run2 (workers=4) |
|---|---|---|
| 블록 시간 | 350s / 50 iter | ~69s / 10 iter |
| **초/iter** | **7.0** | **6.9** |

→ **사실상 동일.** `--num-workers 4`가 속도를 못 줄였다.

### 3.2 왜 안 빨라졌나 — compute-bound

`num_workers`는 **"GPU가 데이터를 기다리며 노는 시간"**을 줄이는 레버다. 그런데 효과가 없다는 건 **GPU가 데이터를 기다린 적이 없다 = 연산 자체로 이미 꽉 찼다**는 뜻이다. 원인은 run1 분석 그대로:
- 풀 k-space 해상도(≈640×368)에서 Restormer(transformer attention) + CSME(BasicUNet)를 **4 cascade** 반복.
- **FP32 강제**(1080 Pascal → BF16/AMP 불가).
- **gradient checkpointing ON**(메모리 아끼려 backward에서 forward 재계산, 시간 +30~50%).

### 3.3 run1 분석의 가설 정정

> run1 분석은 "`num_workers`가 **공짜 개선**"이라고 봤지만, **이번 실측으로 틀렸음이 확인**됐다. 데이터 로딩은 병목이 아니었다. 속도를 줄이려면 **연산량 자체**를 건드려야 한다(§4-B).

그대로면 4651 × 6.9s ≈ **8.9시간/epoch**. iter 2500 인터럽트 = 약 4.8시간 지점.

---

## 4. 다음 액션

### A. "진짜 배우는지" 확정 — 최우선

1. **val 실행해 SSIM_full + SSIM_bbox 측정** (`utils/common/metrics.py`). **bbox가 순위축**이라 제일 중요. 짧게라도(subset/`--max-iters`) 한 번 val까지 완주.
2. **zero-filled 베이스라인 SSIM과 비교** — 네트워크 출력 SSIM이 zero-filled와 비슷하면 아직 실력이 아니다. (§2.4)
3. 학습 loss는 **이동평균(50~100 window)** 으로 그려야 추세가 보인다.

### B. 속도 — `num_workers` 말고 진짜 레버

- **gradient checkpointing OFF** 시도: 이 작은 config는 8GB에 들어갈 수도 있음 → 30~50% 단축 기대.
- **readout crop 640→384가 SDUM 경로에 실제 적용되는지 확인**: 확정 SPEC엔 있지만 어댑터/모델 입력에 실제로 걸려 있는지 미확인. 안 걸려 있으면 공간 연산을 크게 줄일 수 있음. (→ 코드 확인 필요)
- 아이디어 검증은 **`--max-iters` 몇백**으로. 9시간 풀 epoch 돌리며 실험 금지.
- 냉정한 사실: 1080 FP32에서 이 모델은 태생적으로 느리다. 최종 제출 학습엔 **별도 GPU-시간 예산**이 필요.

### C. config / 스케줄 — 후순위

- `[16,32]×4`는 최소급 → **병변 과평활** 우려. 단 **val SSIM_bbox로 파이프라인을 먼저 검증**하고 나서 키울 것.
- loss가 발산하지 않으므로 **LR warmup(Task 5)은 급하지 않다.** 정체를 뚫을 때 재고.

---

## 5. run1 → run2 요약

| 항목 | run1 | run2 (scalefix) | 판정 |
|---|---|---|---|
| 손실 | 1.0 고정 | 0.03~0.28 (SSIM~0.9) | ✅ P0 해결 |
| iter 0 | SSIM 0 | SSIM 0.80 | ✅ 스케일이 원인이었음 확정 |
| 속도 | 7.0s/iter | 6.9s/iter | ❌ 개선 없음(compute-bound) |
| num_workers 가설 | "공짜 개선" 기대 | 효과 없음 | ⚠️ 가설 기각 |
| val SSIM | — | 미측정(인터럽트) | ⬜ 다음 게이트 |

> **성과:** 스케일 정규화로 "학습이 성립"함을 확인. **다음 게이트:** val SSIM_bbox로 "실제로 잘 복원하는지" + 속도 레버(checkpointing/crop) 확보.
