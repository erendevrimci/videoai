import os 
import csv
import subprocess
import json
import shutil
prompt_file = 'video/video-prompt-list.txt'
is_file_created = os.path.exists("video/video-catalog.csv") 

def get_video_metadata(video_path):
    """FFmpeg kullanarak videodan metadata çıkar"""
    try:
        # FFprobe ile metadata çıkarma
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', 
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        # Video akış bilgilerini al
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            print(f"Video akışı bulunamadı: {video_path}")
            return {}
        
        # Metadata çıkar
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        resolution = f"{width}x{height}"
        
        # Aspect ratio hesapla
        if width and height:
            gcd = calc_gcd(width, height)
            aspect_ratio = f"{width//gcd}:{height//gcd}"
        else:
            aspect_ratio = "Unknown"
        
        # FPS hesapla
        fps_str = video_stream.get('avg_frame_rate', '0/0')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = round(num / den, 2) if den != 0 else 0
        else:
            fps = float(fps_str)
        
        duration = f"{round(float(data.get('streams', [{}])[0].get('duration', '')))}s"
         
        
        return {
            'resolution': resolution,
            'aspect_ratio': aspect_ratio,
            'fps': fps,
            'duration': duration,
        }
    
    except Exception as e:
        print(f"Metadata çıkarma hatası ({video_path}): {str(e)}")
        return {}

def calc_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def transfer_video(source_path, destination_folder):
    """Videoyu hedef klasöre taşı"""
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        # Kaynak dosya var mı kontrol et
        if not os.path.exists(source_path):
            print(f"Hata: Kaynak dosya bulunamadı: {source_path}")
            return False
            
        # Dosyayı kopyala
        destination_path = os.path.join(destination_folder, os.path.basename(source_path))
        shutil.copy2(source_path, destination_path)
        print(f"Video taşındı: {destination_path}")
        return True
        
    except Exception as e:
        print(f"Video taşıma hatası ({source_path}): {str(e)}")
        return False

def create_video_catalog():
    headers = ['path','prompt','short_prompt', 'aspect_ratio', 'resolution', 'fps', 'duration']
    
    # UTF-8 kodlaması ile dosyayı aç
    with open(prompt_file, 'r', encoding='utf-8') as file:
        # CSV dosyasını oluştur (eğer yoksa)
        if not is_file_created:
            with open("video/video-catalog.csv", 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        
        # Her satırı oku ve CSV'ye ekle
        for line in file:
            line = line.strip()
            if line and ':' in line:
                parts = line.split(':', 1)  # Sadece ilk iki parçaya böl
                
                if parts[0].strip() == "bos":
                    continue
                if len(parts) == 2:
                    path = f"video/{parts[0].strip()}"
                    prompt = parts[1].strip()
                    if os.path.exists(f"{path}.mp4"):
                        metadata = get_video_metadata(f"{path}.mp4")
                        # transfer_video(f"{path}.mp4", "video/transferred")

                        # Kelime bazlı kısaltma (ilk 30 kelime)
                        words = prompt.split()
                        short_prompt = ' '.join(words[:35]) + ('...' if len(words) > 35 else '')
                        
                        
                        with open("video/video-catalog.csv", 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                path,
                                prompt,
                                short_prompt,    
                                metadata.get('aspect_ratio', ''),
                                metadata.get('resolution', ''),
                                metadata.get('fps', ''),
                                metadata.get('duration', ''),
                                ])

if __name__ == "__main__":
    create_video_catalog()
