from moviepy.editor import VideoFileClip

def extract_wav_from_mp4(mp4_path, wav_path):
    """
    MP4 파일에서 WAV 오디오를 추출하여 저장합니다.

    :param mp4_path: 입력 MP4 파일 경로
    :param wav_path: 출력 WAV 파일 경로
    """
    try:
        # 비디오 파일 로드
        video = VideoFileClip(mp4_path)
        
        # 오디오 추출
        audio = video.audio
        
        if audio is None:
            print("오디오 스트림이 존재하지 않습니다.")
            return
        
        # WAV 파일로 저장
        audio.write_audiofile(wav_path, codec='pcm_s16le')
        
        print(f"WAV 파일이 성공적으로 저장되었습니다: {wav_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 예제 사용법
    
    input_mp4 = "../../../../../Datasets/vggsound_sparse/vggsound_sparse_0VzddQUqVCQ_0_playing_tennis_train_fixed.mp4"    # 입력 MP4 파일 경로
    
    #input_mp4 = "vggsound_sparse__ID2XiPGSEc_0_playing_tennis_train_fixed.mp4"    # 입력 MP4 파일 경로

    output_wav = "vggsound_sparse_0VzddQUqVCQ_0_playing_tennis_train_fixed.wav"  # 출력 WAV 파일 경로
    
    extract_wav_from_mp4(input_mp4, output_wav)
