from pathlib import Path
import os
from typing import Optional, List, Set
from datetime import datetime
import json

class DirectoryExplorer:
    """ディレクトリ構造を探索し、様々な形式で出力するクラス"""
    
    def __init__(self, root_path: str):
        """
        Args:
            root_path: 探索を開始するルートディレクトリのパス
        """
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise ValueError(f"指定されたパスが存在しません: {root_path}")
            
        # 無視するファイルやディレクトリのパターン
        self.ignore_patterns = {
            '.git', '__pycache__', '.ipynb_checkpoints',
            '.pytest_cache', '.vscode', '.idea'
        }

    def _should_ignore(self, path: Path) -> bool:
        """指定されたパスを無視すべきかどうかを判定"""
        return any(pattern in str(path) for pattern in self.ignore_patterns)

    def get_tree_structure(self, max_depth: Optional[int] = None) -> str:
        """ツリー構造を文字列として取得"""
        lines = []
        
        def _explore(path: Path, depth: int, prefix: str = ""):
            if max_depth is not None and depth > max_depth:
                return
                
            if self._should_ignore(path):
                return
                
            # ファイル名またはディレクトリ名を出力
            lines.append(f"{prefix}{'└── ' if prefix else ''}{path.name}")
            
            if path.is_dir():
                # サブディレクトリとファイルを取得してソート
                items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                for item in items:
                    _explore(item, depth + 1, prefix + "    ")
        
        _explore(self.root_path, 0)
        return "\n".join(lines)

    def get_detailed_info(self) -> dict:
        """ディレクトリの詳細情報を辞書形式で取得"""
        info = {
            "total_size": 0,
            "file_count": 0,
            "dir_count": 0,
            "file_types": {},
            "last_modified": None,
            "structure": {}
        }
        
        def _explore(path: Path, current_dict: dict):
            if self._should_ignore(path):
                return
                
            if path.is_file():
                size = path.stat().st_size
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                
                info["total_size"] += size
                info["file_count"] += 1
                
                # ファイル拡張子の統計を更新
                ext = path.suffix.lower()
                info["file_types"][ext] = info["file_types"].get(ext, 0) + 1
                
                # 最終更新日時を更新
                if info["last_modified"] is None or modified > info["last_modified"]:
                    info["last_modified"] = modified
                
                current_dict[path.name] = {
                    "type": "file",
                    "size": size,
                    "modified": modified.isoformat()
                }
            
            elif path.is_dir():
                info["dir_count"] += 1
                current_dict[path.name] = {
                    "type": "directory",
                    "contents": {}
                }
                
                for item in sorted(path.iterdir()):
                    _explore(item, current_dict[path.name]["contents"])
        
        _explore(self.root_path, info["structure"])
        return info

    def save_report(self, output_path: str):
        """結果をJSONファイルとして保存"""
        info = self.get_detailed_info()
        
        # バイト単位のサイズを人間が読みやすい形式に変換
        def _format_size(size: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} TB"
        
        # レポート用に情報を整形
        report = {
            "summary": {
                "total_size": _format_size(info["total_size"]),
                "file_count": info["file_count"],
                "dir_count": info["dir_count"],
                "file_types": info["file_types"],
                "last_modified": info["last_modified"].isoformat() if info["last_modified"] else None
            },
            "structure": info["structure"]
        }
        
        # JSONファイルとして保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

def main():
    """使用例"""
    # 指定されたパスの構造を探索
    explorer = DirectoryExplorer(r"M:\ML\signatejpx")
    
    # ツリー構造を表示
    print("Directory Tree:")
    print(explorer.get_tree_structure())
    print("\n" + "="*50 + "\n")
    
    # 詳細情報を取得して表示
    info = explorer.get_detailed_info()
    print(f"Total Files: {info['file_count']}")
    print(f"Total Directories: {info['dir_count']}")
    print("\nFile Types:")
    for ext, count in info['file_types'].items():
        print(f"  {ext or 'no extension'}: {count}")
    
    # レポートをJSONファイルとして保存
    explorer.save_report("directory_report.json")

if __name__ == "__main__":
    main()