import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import cv2

from models import *
from pytorch_utils import move_data_to_device
import config
from models import ResNet38, Wavegram_Logmel_Cnn14

# CAM 이미지를 저장할 폴더 경로
cam_save_dir = 'cam_images'
os.makedirs(cam_save_dir, exist_ok=True)

# ResNet38 기반 CAM 모델 정의
class ResNet38_CAM(ResNet38):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, device):
        super(ResNet38_CAM, self).__init__(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.device = device
        self.feature_map = None  # feature_map을 저장할 변수 초기화

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        # Spectrogram 추출
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)            # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        
        # feature_map 캡처 (conv_block_after1 이후, 채널 수 = 2048)
        self.feature_map = x.detach()
        
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu_(self.fc1(x))


        embedding = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

    def generate_cam(self, class_idx):
        """
        Class Activation Map을 생성하는 메소드
        """
        # fc_audioset의 가중치 추출
        #print("fc1")
        #print(self.fc1.bias)
        weight_softmax = self.fc_audioset.weight[class_idx].detach().cpu().numpy()  # (2048,)

        weigth_fc1 = self.fc1.weight.detach().cpu().numpy()  # (2048,)

        # fc1이랑 fc_audioset 이랑 w 곱함
        weight_softmax = np.matmul(weight_softmax, weigth_fc1)

        # feature_map의 형태: (batch_size, channels, height, width)
        feature_map = self.feature_map.cpu().numpy()[0]  # (2048, height, width)

        # CAM 생성: weighted sum of feature_map
        cam = np.dot(weight_softmax, feature_map.reshape(feature_map.shape[0], -1))  # (height*width,)
        cam = cam.reshape(feature_map.shape[1], feature_map.shape[2])  # (height, width)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        return cam
    

class WaveGram_CAM(Wavegram_Logmel_Cnn14):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, device):
        super(WaveGram_CAM, self).__init__(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.device = device
        self.feature_map = None  # feature_map을 저장할 변수 초기화

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        #print("a1:", a1.shape)
        a1 = self.pre_block1(a1, pool_size=4)
        #print("a1:", a1.shape)
        a1 = self.pre_block2(a1, pool_size=4)
        #print("a1:", a1.shape)
        a1 = self.pre_block3(a1, pool_size=4)
        #print("a1:", a1.shape)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        #print("a1:", a1.shape)
        a1 = self.pre_block4(a1, pool_size=(2, 1))
        #print("a1:", a1.shape)

        # Log mel spectrogram
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        
        #print("x:",x.shape)
        #print("a1:", a1.shape)
        
        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        # feature_map 캡처 (conv_block_after1 이후, 채널 수 = 2048)
        self.feature_map = x.detach()
        
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu_(self.fc1(x))
        embedding = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

    def generate_cam(self, class_idx):
        """
        Class Activation Map을 생성하는 메소드
        """
        # fc_audioset의 가중치 추출
        #print("fc1")
        #print(self.fc1.bias)
        weight_softmax = self.fc_audioset.weight[class_idx].detach().cpu().numpy()  # (2048,)

        weigth_fc1 = self.fc1.weight.detach().cpu().numpy()  # (2048,)

        # fc1이랑 fc_audioset 이랑 w 곱함
        weight_softmax = np.matmul(weight_softmax, weigth_fc1)

        # feature_map의 형태: (batch_size, channels, height, width)
        feature_map = self.feature_map.cpu().numpy()[0]  # (2048, height, width)

        # CAM 생성: weighted sum of feature_map
        cam = np.dot(weight_softmax, feature_map.reshape(feature_map.shape[0], -1))  # (height*width,)
        cam = cam.reshape(feature_map.shape[1], feature_map.shape[2])  # (height, width)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        return cam

def save_combined_image(spectrogram, cam, save_dir, class_name):
    """
    Spectrogram, CAM, 그리고 Overlay를 하나의 이미지 파일로 저장하는 함수
    """
    # Spectrogram을 [0, 255] 범위로 스케일링


    # CAM을 Spectrogram 크기에 맞게 조정
    
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)


    # Spectrogram을 3채널으로 변환 (BGR)
    if len(spectrogram.shape) == 2:
        spectrogram_bgr = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)
    else:
        spectrogram_bgr = spectrogram

    # Overlay 생성: 두 이미지의 채널 수가 동일해야 함
    overlay = cv2.addWeighted(spectrogram_bgr, 0.7, heatmap, 0.3, 0)

    # BGR을 RGB로 변환 (Matplotlib 호환)
    spectrogram_rgb = cv2.cvtColor(spectrogram_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Matplotlib을 사용하여 세 개의 서브플롯을 세로로 배열
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # 세로로 3개

    # Spectrogram 표시
    axs[0].imshow(spectrogram_rgb, aspect='auto', origin='lower')
    axs[0].set_title('Spectrogram')
    axs[0].axis('off')

    # CAM 표시
    #im = axs[1].imshow(heatmap_rgb, aspect='auto', origin='lower')
    axs[1].imshow(heatmap_rgb, aspect='auto', origin='lower')
    axs[1].set_title('CAM (Color indicates intensity)')
    axs[1].axis('off')

    # Colorbar 추가
    #cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    #cbar.set_label('CAM Intensity', rotation=270, labelpad=15)

    # Overlay 표시
    axs[2].imshow(overlay_rgb, aspect='auto', origin='lower')
    axs[2].set_title('Overlay')
    axs[2].axis('off')

    plt.tight_layout()

    # 이미지 저장
    #if not os.path.exists(combined_img_path):
    #    os.makedirs(combined_img_path)
    combined_img_path = os.path.join(save_dir, f"Combined_{class_name}.jpg")

    plt.savefig(combined_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Combined image saved at {combined_img_path}")

def save_class_time_cam_image(aggregated_cams_array, class_names, save_dir):
    """
    클래스(y축)와 시간(x축)을 갖는 CAM 값을 저장하는 함수
    """

    fig, ax = plt.subplots(figsize=(12, 2 + len(class_names) * 0.5))
    im = ax.imshow(aggregated_cams_array, aspect='auto', origin='upper', cmap='jet', interpolation='nearest')

    for idx in range(1, len(class_names)):
        ax.hlines(y=idx - 0.5, xmin=0, xmax=aggregated_cams_array.shape[1] - 1, colors='white', linestyles='-', linewidth=1.0)


    ax.set_yticks(np.arange(len(class_names))) 
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Time Frames')
    ax.set_title('Aggregated CAM over Time for Top Classes')

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Aggregated_CAM_Top_Classes.jpg")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Aggregated CAM image saved at {save_path}")



def audio_tagging(args):
    """오디오 클립의 태깅 및 CAM 생성 결과를 추론합니다."""
    # 인자 설정
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    print("classes_num:", classes_num)

    # 모델 초기화
    
    Model = ResNet38_CAM  # 직접 참조
    #Model = WaveGram_CAM
    model = Model(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, 
                  mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num, device=device)

    # 사전 학습된 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)  # 모델을 디바이스로 이동

    # GPU 설정
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        print('Using GPU')

    model.eval()  # 평가 모드로 전환


    # 오디오 로드
    waveform, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform[None, :]  # (1, audio_length)
    waveform = move_data_to_device(waveform, device)


    # 추론
    with torch.no_grad():
        output = model(waveform)

    # 상위 k개 클래스 예측
    top_k = 11
    clipwise_output = output['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    print("Top classes with probabilities:")
    for k in range(top_k):
        class_idx = sorted_indexes[k]
        print('{}: {:.3f}'.format(np.array(labels)[class_idx], clipwise_output[class_idx]))

    # Spectrogram 추출 (CAM 시각화를 위해)
    spectrogram = model.logmel_extractor(model.spectrogram_extractor(waveform)).detach().cpu().numpy()[0][0]
    print(f"Spectrogram shape: {spectrogram.shape}") 
    # 필요시 전치 (예: spectrogram = spectrogram.T)

    aggregated_cams = []
    class_names = []


    # CAM 생성 및 시각화
    for k in range(top_k):
        class_idx = sorted_indexes[k]
        class_name = np.array(labels)[class_idx]  # 클래스 이름으로 CAM 저장
        cam = model.generate_cam(class_idx)
        cam = cam.T

        spectrogram_scaled = spectrogram - np.min(spectrogram)
        spectrogram_scaled = spectrogram_scaled / np.max(spectrogram_scaled)
        spectrogram_scaled = np.uint8(255 * spectrogram_scaled)
        spectrogram_scaled = spectrogram_scaled.T

        cam = cv2.resize(cam, (spectrogram_scaled.shape[1], spectrogram_scaled.shape[0]))
        save_combined_image(spectrogram_scaled, cam, cam_save_dir, class_name)  # 수정된 함수 호출

        # class 마다 확률 곱하기
        class_idx = sorted_indexes[k]
        #print(clipwise_output[class_idx])
        cam = cam * clipwise_output[class_idx]
        #주파수 축으로 CAM 압축
        cam_aggregated = np.mean(cam, axis=0)  # np.sum(cam, axis=0) 또는 np.mean(cam, axis=0)
        aggregated_cams.append(cam_aggregated)
        class_names.append(class_name)

    # CAM 값을 2D 배열로 변환 (클래스 수 x 시간)
    aggregated_cams_array = np.stack(aggregated_cams, axis=0)  # Shape: (k, time_steps)
    # 클래스-시간 CAM 이미지를 저장하는 함수 호출
    save_class_time_cam_image(aggregated_cams_array, class_names, cam_save_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Tagging and CAM Generation')

    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, default="ResNet38_CAM")
    #parser.add_argument('--checkpoint_path', type=str, default="C:/Users/Noah/Desktop/MMG/Codes/MMG/CAM/audioset_tagging_cnn/pytorch/Wavegram_Logmel_Cnn14_mAP=0.439.pth")
    parser.add_argument('--checkpoint_path', type=str, default="C:/Users/Noah/Desktop/MMG/Codes/MMG/CAM/audioset_tagging_cnn/pytorch/ResNet38_mAP=0.434.pth")
    parser.add_argument('--audio_path', type=str, default="C:/Users/Noah/Desktop/MMG/Codes/MMG/CAM/audioset_tagging_cnn/pytorch/bus_chatter.wav")
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    audio_tagging(args)
