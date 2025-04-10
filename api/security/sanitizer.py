import re

import hashlib
import time
from typing import Union, Dict, Any, List, Optional





SQL_KEYWORDS = [
    # Temel SQL komutları
    r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bUNION\b', r'\bINSERT\b', 
    r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b', r'\bCREATE\b', r'\bALTER\b', 
    r'\bTRUNCATE\b', r'\bRENAME\b', r'\bEXEC\b', r'\bEXECUTE\b', r'\bMERGE\b',
    r'\bGRANT\b', r'\bREVOKE\b', r'\bCOMMIT\b', r'\bROLLBACK\b', r'\bSAVEPOINT\b',
    
    # Tablo ve veri işleme
    r'\bJOIN\b', r'\bINNER\b', r'\bOUTER\b', r'\bLEFT\b', r'\bRIGHT\b', r'\bFULL\b', 
    r'\bCROSS\b', r'\bORDER\s+BY\b', r'\bGROUP\s+BY\b', r'\bHAVING\b', r'\bLIMIT\b',
    r'\bOFFSET\b', r'\bTOP\b', r'\bDISTINCT\b', r'\bALL\b', r'\bAS\b', r'\bIN\b',
    r'\bBETWEEN\b', r'\bEXISTS\b', r'\bCASE\b', r'\bWHEN\b', r'\bTHEN\b', r'\bELSE\b',
    
    # Fonksiyonlar
    r'\bCOUNT\b', r'\bSUM\b', r'\bMAX\b', r'\bMIN\b', r'\bAVG\b', r'\bCONCAT\b',
    r'\bSUBSTRING\b', r'\bCHAR\b', r'\bHEX\b', r'\bUNHEX\b', r'\bLENGTH\b', r'\bTRIM\b',
    
    # Operatörler
    r'\bOR\b', r'\bAND\b', r'\bNOT\b', r'\bXOR\b', r'\bIS\s+NULL\b', r'\bIS\s+NOT\s+NULL\b',
    r'\bLIKE\b', r'\bREGEXP\b', r'\bSIMILAR\s+TO\b',
    
    # Sistem fonksiyonları
    r'\bVERSION\b', r'\bDATABASE\b', r'\bSCHEMA\b', r'\bIF\b', r'\bIFNULL\b', 
    r'\bCONVERT\b', r'\bCAST\b', r'\bCOALESCE\b',
    
    # Yorum satırları
    r'--', r'/\*', r'\*/'
]

# SQL Injection saldırı kalıpları
ATTACK_PATTERNS = [
    # Boolean-based
    r"'\s*OR\s+'1'\s*=\s*'1", 
    r'"\s*OR\s+"1"\s*=\s*"1',
    r"'\s*OR\s+1\s*=\s*1",
    r'"\s*OR\s+1\s*=\s*1',
    r"'\s*OR\s+1\s*=\s*'1",
    r'"\s*OR\s+1\s*=\s*"1',
    r"'\s*OR\s+'1'\s*=\s*1",
    r'"\s*OR\s+"1"\s*=\s*1',
    
    # UNION based
    r"'\s*UNION\s+ALL\s+SELECT\s+",
    r'"\s*UNION\s+ALL\s+SELECT\s+',
    r"'\s*UNION\s+SELECT\s+",
    r'"\s*UNION\s+SELECT\s+',
    
    # Stacked queries
    r"';\s*SELECT\s+",
    r'";\s*SELECT\s+',
    r"';\s*DROP\s+",
    r'";\s*DROP\s+',
    r"';\s*INSERT\s+",
    r'";\s*INSERT\s+',
    r"';\s*UPDATE\s+",
    r'";\s*UPDATE\s+',
    r"';\s*DELETE\s+",
    r'";\s*DELETE\s+',
    
    # Yorum satırları ile sonlandırma
    r"'\s*--",
    r'"\s*--',
    r"'\s*/\*",
    r'"\s*/\*',
    
    # Sürekli ifadeler
    r"'\s*OR\s*'x'='x",
    r'"\s*OR\s*"x"="x',
    
    # SQL Server özel saldırıları
    r"'\s*;\s*WAITFOR\s+DELAY\s+",
    r'"\s*;\s*WAITFOR\s+DELAY\s+',
    r"'\s*;\s*EXEC\s+",
    r'"\s*;\s*EXEC\s+',
    
    # PostgreSQL özel saldırıları
    r"'\s*;\s*SELECT\s+pg_sleep",
    r'"\s*;\s*SELECT\s+pg_sleep',
    
    # MySQL özel saldırıları
    r"'\s*;\s*SELECT\s+SLEEP",
    r'"\s*;\s*SELECT\s+SLEEP',
    
    # NoSQL injection
    r"\{\s*\$where\s*:\s*",
    r"\{\s*\$regex\s*:\s*",
    r"\{\s*\$ne\s*:\s*",
    r"\{\s*\$gt\s*:\s*",
    r"\{\s*\$lt\s*:\s*",
]

# Özel karakterler için beyaz liste
ALLOWED_CHARS_PATTERN = r'^[a-zA-Z0-9_\-. ]+$'

# Karakter kodlama ve escape girişimleri
ENCODING_PATTERNS = [
    r'%27', r'%22', r'%5C',  # URL Encoding: ', ", \
    r'\\u00', r'\\x',        # Unicode ve Hex escapes
    r'&#',                   # HTML entity encoding
    r'char\(', r'chr\(',     # CHAR() ve CHR() fonksiyonları
]

# Güvenlik ihlali şüphelerini izlemek için sözlük
security_violations = {}

class SQLInjectionError(Exception):
    """SQL Injection tespit edildiğinde fırlatılan özel hata sınıfı"""
    pass


def contains_sql_keywords(input_value: str) -> bool:
    """Giriş değerinde SQL anahtar kelimeleri olup olmadığını kontrol eder."""
    # WhiteSpace ve yorum satırlarını normalize et
    normalized = re.sub(r'\s+', ' ', input_value.lower())
    
    # Olası kodlama girişimlerini kontrol et
    for pattern in ENCODING_PATTERNS:
        if re.search(pattern, normalized):
            return True
    
    # SQL anahtar kelimeleri kontrol et
    for keyword in SQL_KEYWORDS:
        if re.search(keyword, normalized, re.IGNORECASE):
            return True
    
    return False

def contains_attack_patterns(input_value: str) -> Optional[str]:
    """Giriş değerinde bilinen saldırı kalıpları olup olmadığını kontrol eder."""
    # Boşlukları normalize et
    normalized = re.sub(r'\s+', ' ', input_value.lower())
    
    for pattern in ATTACK_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return pattern
    
    return None

def is_safe_input(input_value: str) -> bool:
    """Bir girişin güvenli olup olmadığını kontrol eder (beyaz liste yaklaşımı)."""
    return bool(re.match(ALLOWED_CHARS_PATTERN, input_value))

def sanitize_sql_param(input_value: str) -> str:
    """SQL sorgularında kullanılacak parametreler için özel temizleme fonksiyonu."""
    if not input_value or not isinstance(input_value, str):
        return ""
    
    # Tehlikeli karakterleri kaldır
    sanitized = re.sub(r'[;\'"\\\(\)=]', '', input_value)
    
    # Yorum satırlarını kaldır
    sanitized = re.sub(r'--.*$', '', sanitized)
    sanitized = re.sub(r'/\*.*?\*/', '', sanitized)
    
    # Birden fazla boşluğu tek boşluğa dönüştür
    sanitized = re.sub(r'\s+', ' ', sanitized.strip())
    
    return sanitized

def sanitize_input(
    input_value: str, 
    strict_mode: bool = True, 
    context: str = "general",
    ip_address: Optional[str] = None, 
    user_id: Optional[str] = None
) -> str:
    """
    Gelişmiş SQL enjeksiyon koruması ile kullanıcı girdisini temizler.
    
    Args:
        input_value: Temizlenecek kullanıcı girdisi
        strict_mode: True ise daha katı kontroller uygulanır
        context: Girdinin kullanılacağı bağlam (genel, id, search, script vb.)
        ip_address: İsteği yapan kullanıcının IP adresi (opsiyonel)
        user_id: İsteği yapan kullanıcının kimliği (opsiyonel)
        
    Returns:
        Temizlenmiş girdi
        
    Raises:
        SQLInjectionError: Potansiyel SQL enjeksiyon girişimi tespit edilirse
    """
    if not isinstance(input_value, str):
        return input_value
    
    # Boş girdi kontrolü
    if not input_value.strip():
        return ""
    
    # Kontekste göre farklı güvenlik seviyeleri uygula
    if context == "id" or context == "numeric":
        # ID ve numerik değerler için sadece sayılar kabul edilir
        if not input_value.isdigit():
            
            raise SQLInjectionError(f"Geçersiz {context} formatı")
        return input_value
    
    elif context == "email":
        # E-posta adresleri için basit doğrulama
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', input_value):
            
            raise SQLInjectionError("Geçersiz e-posta formatı")
        return input_value
    
    elif context == "script":
        # Script metinleri için sadece basit tehlikeli karakter temizliği yap,
        # SQL anahtar kelime kontrolü yapma çünkü doğal dilde kullanılan 
        # "where", "then", "case" gibi kelimeler sıkça kullanılır
        sanitized = re.sub(r'[;\'"\\]', '', input_value)
        return sanitized
    
    # Bilinen saldırı kalıplarını kontrol et
    attack_pattern = contains_attack_patterns(input_value)
    if attack_pattern:
        
        raise SQLInjectionError("Potansiyel SQL injection tespit edildi")
    
    # SQL anahtar kelimelerini kontrol et
    if contains_sql_keywords(input_value):
        
        raise SQLInjectionError("SQL komutu içeren sorgu tespit edildi")
    
    # Strict modda beyaz liste kontrolü yap
    if strict_mode and not is_safe_input(input_value):
        
        raise SQLInjectionError("İzin verilmeyen karakterler içeriyor")
    
    # Bağlama göre uygun temizleme uygula
    if context == "search":
        # Arama sorguları için daha esnek temizleme
        sanitized = sanitize_sql_param(input_value)
        # En az 3 karakter kontrolü
        if len(sanitized) < 3:
            return ""
        return sanitized
    else:
        # Genel amaçlı temizleme
        sanitized = re.sub(r'[;\'"\\]', '', input_value)
        return sanitized

def sanitize_dict(
    data: Dict[str, Any], 
    strict_mode: bool = True,
    ip_address: Optional[str] = None, 
    user_id: Optional[str] = None,
    is_script_content: bool = False
) -> Dict[str, Any]:
    """
    Bir sözlükteki tüm string değerleri temizler.
    
    Args:
        data: Temizlenecek sözlük verisi
        strict_mode: True ise daha katı kontroller uygulanır
        ip_address: İsteği yapan kullanıcının IP adresi
        user_id: İsteği yapan kullanıcının kimliği
        is_script_content: True ise, script içeriği için özel işlem yapılır
        
    Returns:
        Temizlenmiş sözlük
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Script içeriği için özel işlem
            if is_script_content and key == "script":
                result[key] = sanitize_input(value, context="script", ip_address=ip_address, user_id=user_id)
            else:
                # Alan adına göre bağlam seç
                context = "general"
                if key in ["email", "mail"]:
                    context = "email"
                elif key in ["id", "user_id", "item_id"]:
                    context = "id"
                elif key in ["search", "query", "q", "keyword"]:
                    context = "search"
                    
                # Uygun bağlamda sanitize et
                result[key] = sanitize_input(
                    value, 
                    strict_mode=strict_mode, 
                    context=context,
                    ip_address=ip_address, 
                    user_id=user_id
                )
        elif isinstance(value, dict):
            # İç içe sözlükleri de temizle
            result[key] = sanitize_dict(
                value, 
                strict_mode=strict_mode, 
                ip_address=ip_address, 
                user_id=user_id,
                is_script_content=is_script_content
            )
        elif isinstance(value, list):
            # Listelerdeki her öğeyi temizle
            sanitized_list = []
            for item in value:
                if isinstance(item, dict):
                    sanitized_item = sanitize_dict(
                        item, 
                        strict_mode=strict_mode, 
                        ip_address=ip_address, 
                        user_id=user_id,
                        is_script_content=is_script_content
                    )
                    sanitized_list.append(sanitized_item)
                elif isinstance(item, str):
                    sanitized_item = sanitize_input(
                        item, 
                        strict_mode=strict_mode, 
                        context="general",
                        ip_address=ip_address, 
                        user_id=user_id
                    )
                    sanitized_list.append(sanitized_item)
                else:
                    sanitized_list.append(item)
            result[key] = sanitized_list
        else:
            # Diğer tipteki değerleri doğrudan kullan
            result[key] = value
            
    return result

