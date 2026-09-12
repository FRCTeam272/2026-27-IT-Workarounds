import subprocess
import time

# PowerShell script to check and enable Windows Mobile Hotspot
PS_HOTSPOT_COMMAND = """
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

Function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

[Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime] | Out-Null

# Get default connection profile for internet sharing
$connectionProfile = [Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime]::GetInternetConnectionProfile()
$tetheringManager = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($connectionProfile)

if ($tetheringManager.TetheringOperationalState -ne 1) {
    Write-Output "Hotspot is OFF. Turning ON..."
    $result = Await ($tetheringManager.StartTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    Write-Output "Result: $($result.Status)"
} else {
    Write-Output "Hotspot is already ON."
}
"""

def keep_hotspot_alive(check_interval_seconds=15):
    """Periodically checks and turns on the Windows Mobile Hotspot."""
    print("Starting Windows Hotspot keeper. Press Ctrl+C to stop.")
    while True:
        try:
            process = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", PS_HOTSPOT_COMMAND],
                capture_output=True,
                text=True,
                check=False
            )
            output = process.stdout.strip()
            if output:
                print(f"[{time.strftime('%X')}] {output}")
        except Exception as e:
            print(f"Error executing check: {e}")

        time.sleep(check_interval_seconds)

if __name__ == "__main__":
    keep_hotspot_alive(15)