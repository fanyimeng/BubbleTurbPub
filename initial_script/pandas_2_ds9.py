import pandas as pd

# Define the function to convert each row to a DS9 region string


def row_to_ds9_region(row):
    return f"ellipse({row['ra_hms']},{row['dec_dms']},{row['maj_as']}\",{row['min_as']}\",{row['pa']}) # text={{{row['id']}}}"


# Path to the ASCII file containing the table
ascii_file_path = "0709.tab"

# Read the file into a DataFrame
# Make sure to replace '/path/to/your/table_data.txt' with the actual file path
df = pd.read_fwf(ascii_file_path, header=0, infer_nrows=int(1e6))

# Apply the function to each row to create DS9 regions
ds9_regions = df.apply(row_to_ds9_region, axis=1)

# Save the regions to a new DS9 region file
ds9_file_path = "0709.reg"
with open(ds9_file_path, "w") as file:
    file.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    file.write("fk5\n")
    for region in ds9_regions:
        file.write(region + "\n")
