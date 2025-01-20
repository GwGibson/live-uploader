# Not really part of the uploader but a useful utility function that can be expanded on

def generate_channel_mapping(sensor_start_number: int, num_channels: int) -> str:
    mappings = []
    for i in range(num_channels):
        channel_number = i + 1
        sensor_number = sensor_start_number + i
        mappings.append(f"{channel_number}:{sensor_number}")

    return ", ".join(mappings)


def main():
    result = generate_channel_mapping(1149, 428)
    # Probably better to copy to clipboard or create a file?
    # Ultimately, this should be done in a more user-friendly way
    print(result)


if __name__ == "__main__":
    main()
