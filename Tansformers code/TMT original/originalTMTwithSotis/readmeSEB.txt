python train_tilt_static.py --train_path /home/sebastian/sebas/TMT-main/datasetStatic/syn_static/train_turb --val_path /home/sebastian/sebas/TMT-main/datasetStatic/syn_static/test_turb --log_path /home/sebastian/sebas/TMT-main/logs


python train_tilt_dynamic.py --train_path /home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train --val_path /home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/test --log_path /home/sebastian/sebas/TMT-main/logs


python train_TMT_dynamic_2stage.py --path_tilt /home/sebastian/sebas/TMT-main/logs/dynamic-tilt_06-11-2024-15-42-42/checkpoints/model_last.pth --train_path /home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train --val_path /home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/test --log_path /home/sebastian/sebas/TMT-main/logsBlur --run_name "tuleperaconlapapaya"


esta carpeta entrena la red usando train_TMT_static.py esta cuenta con las dos partes de la red y en esta carpeta se entrenausando el dataset otis, se puede
cambiar el numero de frames hasta 50, tal vez 49 aprovechando la cantidad de frames que teneemos por cada valor de la constante c2
los frames de blur podrian ser 80 pero hassta ahora se utiliza la misma cantidad e frames para ambos casos.
miento no se pueden poner mas de 14 frames a la vez porque salta un error que no quiero mirar haha


se esta corriendoc on el comando:

python train_TMT_static.py --train_path /home/sebastian/sebas/TMT-originalSotis/datasetStatic/syn_static/train_turb --val_path /home/sebastian/sebas/TMT-originalSotis/datasetStatic/syn_static/test_turb --log_path /home/sebastian/sebas/TMT-originalSotis/logs --run_name entrenadoConFramesOrdenadosPorC2Tilt 

