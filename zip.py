import os
import zipfile

def convert_csv_to_zip(input_directory):
    """
    Convert all CSV files in the specified directory to ZIP files.
    
    Args:
        input_directory (str): Path to the directory containing CSV files
    """
    # Ensure the input directory exists
    if not os.path.exists(input_directory):
        print(f"エラー: ディレクトリが存在しません - {input_directory}")
        return

    # Create output directory if it doesn't exist
    output_directory = os.path.join(input_directory, 'zipped_csvs')
    os.makedirs(output_directory, exist_ok=True)

    # Counter for successful and failed conversions
    successful_conversions = 0
    failed_conversions = 0

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a CSV
        if filename.lower().endswith('.csv'):
            try:
                # Full path to the input CSV file
                input_filepath = os.path.join(input_directory, filename)
                
                # Create ZIP filename (same name as CSV but with .zip extension)
                zip_filename = os.path.splitext(filename)[0] + '.zip'
                output_filepath = os.path.join(output_directory, zip_filename)
                
                # Create a ZIP file and add the CSV
                with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(input_filepath, arcname=filename)
                
                print(f"変換完了: {filename} → {zip_filename}")
                successful_conversions += 1
            
            except Exception as e:
                print(f"変換エラー: {filename} - {str(e)}")
                failed_conversions += 1

    # Print summary
    print("\n変換サマリ:")
    print(f"総ファイル数: {successful_conversions + failed_conversions}")
    print(f"成功: {successful_conversions}")
    print(f"失敗: {failed_conversions}")
    print(f"出力ディレクトリ: {output_directory}")

# Usage example
if __name__ == "__main__":
    # Replace this path with your actual directory path
    input_dir = r"M:\ML\signatejpx\output\processed"
    convert_csv_to_zip(input_dir)