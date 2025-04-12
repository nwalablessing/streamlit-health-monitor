import xml.etree.ElementTree as ET
import pandas as pd

def parse_health_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []

    for record in root.findall("Record"):
        rtype = record.attrib.get("type")
        if rtype in [
            "HKQuantityTypeIdentifierHeartRate",
            "HKQuantityTypeIdentifierOxygenSaturation",
            "HKQuantityTypeIdentifierBodyMass"
        ]:
            records.append({
                "type": rtype.split("Identifier")[-1],
                "value": float(record.attrib.get("value", 0)),
                "unit": record.attrib.get("unit"),
                "start_date": record.attrib.get("startDate"),
            })

    df = pd.DataFrame(records)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

if __name__ == "__main__":
    df = parse_health_data("export.xml")
    df.to_csv("apple_health.csv", index=False)
    df.to_json("apple_health.json", orient="records", indent=2)
    print("✅ export.xml converted to apple_health.csv and apple_health.json")
