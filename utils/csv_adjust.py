import pandas as pd 
import json
import subprocess

analyzed_csv_path = "video/video-catalog-analyzed-not_filtered.csv"
adjusted_csv_path = "video/video-catalog-adjusted-not_filtered.csv"


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


def get_metadata(row):
    metadata = get_video_metadata(row)
    return metadata
    

def main():
    filtered_csv = pd.read_csv(analyzed_csv_path)
    filtered_csv = filtered_csv.drop(['aspect_ratio','resolution','fps','duration'],axis=1)
    extracted_metadata = filtered_csv['path'].apply(lambda row: get_metadata(row))
    extracted_metadata_df = pd.DataFrame(extracted_metadata.to_list())
    final_df = pd.concat([filtered_csv,extracted_metadata_df],axis=1)
    final_df = final_df.reset_index(drop=True)
    final_df.to_csv(adjusted_csv_path,index=False)
    


main()