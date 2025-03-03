import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
import joblib

class FeatureExtractor:
    def __init__(self):
        self.feature_names = []
    
    def extract_color_histogram(self, image, bins=32):
        """提取颜色直方图特征"""
        # 转换为HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 计算每个通道的直方图
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # 归一化
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v])

    def extract_color_moments(self, image):
        """提取颜色矩特征"""
        # 转换为浮点型
        image = image.astype(float)
        
        # 计算每个通道的矩
        moments = []
        for i in range(3):
            channel = image[:,:,i]
            mean = np.mean(channel)
            std = np.std(channel)
            skew = np.mean((channel - mean) ** 3) / (std ** 3) if std > 0 else 0
            moments.extend([mean, std, skew])
            
        return np.array(moments)

    def extract_texture_glcm(self, image, distances=[1], angles=[0]):
        """提取GLCM纹理特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算GLCM矩阵
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # 计算GLCM属性
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        
        return np.array([contrast[0,0], dissimilarity[0,0], homogeneity[0,0], 
                        energy[0,0], correlation[0,0]])

    def extract_basic_stats(self, image):
        """提取基本统计特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算统计量
        mean = np.mean(gray)
        std = np.std(gray)
        # 计算熵
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist.flatten() / hist.sum()
        ent = entropy(hist[hist > 0])
        
        return np.array([mean, std, ent])

    def extract_all_features(self, image):
        """提取所有特征"""
        features = []
        
        # 颜色直方图
        color_hist = self.extract_color_histogram(image)
        features.extend(color_hist)
        
        # 颜色矩
        color_moments = self.extract_color_moments(image)
        features.extend(color_moments)
        
        # GLCM纹理特征
        glcm_features = self.extract_texture_glcm(image)
        features.extend(glcm_features)
        
        # 基本统计特征
        basic_stats = self.extract_basic_stats(image)
        features.extend(basic_stats)
        
        return np.array(features)

    def calculateFeat(self, id, img):
        return id, self.extract_all_features(img)
    
    def __call__(self, img_list):
        id_list = list(range(len(img_list)))
        results = joblib.Parallel(n_jobs=-1, backend='multiprocessing', verbose=0)(
            joblib.delayed(self.calculateFeat)(img_id, img) for img_id, img in zip(id_list, img_list)
        )
        # 对结果按id排序
        sorted_results = sorted(results, key=lambda x: x[0])
        # 提取特征向量
        features = [feat for _, feat in sorted_results]
        return features

def compute_similarity(features1, features2, method='cosine'):
    """计算特征向量之间的相似度"""
    if method == 'cosine':
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    elif method == 'euclidean':
        return np.linalg.norm(features1 - features2)
    else:
        raise ValueError("Unsupported similarity method")

# 使用示例
if __name__ == "__main__":
    # 读取图像
    image_path = "example.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read image")
    else:
        # 创建特征提取器
        extractor = FeatureExtractor()
        
        # 提取特征
        features = extractor.extract_all_features(image)
        
        print(f"提取的特征维度: {len(features)}")
        
        # 如果要比较两张图片的相似度
        image2_path = "example2.jpg"
        image2 = cv2.imread(image2_path)
        if image2 is not None:
            features2 = extractor.extract_all_features(image2)
            similarity = compute_similarity(features, features2)
            print(f"两张图片的余弦相似度: {similarity}")
