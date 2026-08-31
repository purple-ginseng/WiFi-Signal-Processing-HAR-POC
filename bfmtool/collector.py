import paramiko
import getpass
import argparse 
import time
import threading
import os

import datetime

class BFMCollector:
    """
    A class to connect to an OpenWrt router via SSH,
    run commands, and collect the output.
    """
    def __init__(
            self, 
            host, 
            username, 
            password=None, 
            port=22, 
            local_pcap_dir="bfm_pcap", 
            filename='bfm', 
            filesize=5,
            filecount=2
        ):
        """
        Initializes the collector with connection details.

        Args:
            host (str): The IP address or hostname of the router.
            username (str): The username for SSH login (e.g., 'root').
            password (str, optional): The password for the user.
            port (int, optional): The SSH port. Defaults to 22.
            local_pcap_dir (str, optional): Directory where local pcaps are stored.
            filename (str | Callable, optional): Base filename (or callable returning one).
            filesize (int, optional): Maximum size in MB before tcpdump rotates files.
            filecount (int, optional): Number of rotating capture files to keep on the router.
        """
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.client = None 
        
        self.filesize = filesize
        self.filename = filename
        self.filecounter = -1
        self.local_pcap_dir = local_pcap_dir
        self.filecount = max(1, int(filecount))
        self.pcap_collector_thread = None
        self._stop_event = threading.Event()
        self._processed_files = set() 
        self._collected_files = set()
        
        self._is_stopping = False
        self._is_fast = False

    def get_filename(self):
        self.filecounter += 1
        if callable(self.filename):
            name = self.filename()
            if name.endswith('.pcap'):
                name = name[:-5]
            # The counter matters here too: caller-supplied names are usually
            # timestamped only to the second, so two chunks collected within the
            # same second (both rotation files in the final sweep, say) would
            # otherwise resolve to the same local path and overwrite each other.
            return f"{name}_{self.filecounter}.pcap"
        else:
            if self.filename.endswith('.pcap'):
                return f"{self.filename[:-5]}{self.filecounter}.pcap"
            else:
                return f"{self.filename}{self.filecounter}.pcap"
    
    def connect(self):
        """Establishes the SSH connection."""
        if self.client is not None:
            print("Already connected.")
            return

        try:
            # If no password was provided, prompt for it securely
            if self.password is None:
                self.password = getpass.getpass(
                    f"Enter password for {self.username}@{self.host}: "
                )

            # Create an SSH client instance
            self.client = paramiko.SSHClient()
            # Automatically add the server's host key (less secure, but fine for local devices)
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            print(f"Connecting to {self.host}...")
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=10 # Add a timeout for robustness
            )
            print(f"✅ Connection successful!")

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.client = None # Ensure client is reset on failure
            # Re-raise the exception to stop the 'with' block
            raise

    def run_command(self, command):
        """
        Runs a command on the remote router and returns the output.

        Args:
            command (str): The command string to execute.

        Returns:
            tuple: A tuple containing (stdout, stderr).
                   Returns (None, None) if not connected.
        """
        if not self.client:
            print("❌ Cannot run command. Client is not connected.")
            return None, None

        print(f"Executing command: '{command}'")
        stdin, stdout, stderr = self.client.exec_command(command)

        # Read the output and error streams and decode them from bytes to string
        output_str = stdout.read().decode('utf-8').strip()
        error_str = stderr.read().decode('utf-8').strip()

        return output_str, error_str

    def run_iperf3(self):
        self.run_command('iperf3 -s > /dev/null 2>&1 &')
    
    def kill_iperf3(self):
        self.run_command('killall -9 iperf3')

    def fast(self):
        self._is_fast = True
    
    def slow(self):
        self._is_fast = False

    def run_tcpdump(self, start_collector=True):
        """
        Starts tcpdump to capture packets on the remote device.

        If the command fails (e.g., mon0 interface is down), it attempts
        to reset the interface and tries to start the capture again.
        
        Args:
            filename (str): The name for the capture file on the remote device,
                            which will be saved in the /tmp/ directory.
        """
        # Define the remote file path and the tcpdump command.
        # We run it in the background (&) and redirect output to prevent blocking.
        file_prefix = "bfm_capture"
        remote_path_prefix = f"/tmp/{file_prefix}"

        tcpdump_started = False

        if not self._is_fast:
            command = (
                f"tcpdump -i mon0 -p -w {remote_path_prefix} "
                f"-W {self.filecount} -C {self.filesize} "
                "'wlan[24] == 21' > /dev/null 2>&1 &"
            )
        else:
            command = f"tcpdump -i mon0 -w {remote_path_prefix} -G 1 'wlan[24] == 21' > /dev/null 2>&1 &"

        print("Attempting to start tcpdump...")
        # The run_command for a backwground process should return instantly with no error.
        _, initial_error = self.run_command(f"ps | grep '[t]cpdump -i mon0'")
        if initial_error:
            print(f"Could not run initial check. Error: {initial_error}")

        # For this to work, we need to check if an error occurred starting the process
        # The most robust way is to check if the process is running after we start it.
        # First, let's kill any old instances.
        self.run_command("killall tcpdump")
        
        print("Cleaning up old capture files...")
        cleanup_command = f"rm -f /tmp/{file_prefix}*"
        self.run_command(cleanup_command)

        # Now, run the command
        self.run_command(command)
        time.sleep(1) # Give it a moment to start up or fail

        # Check if the process is running
        output, _ = self.run_command("pgrep -f 'tcpdump -i mon0'")

        # If 'pgrep' returns no output, the command failed to start
        if not output:
            print("⚠️ tcpdump failed to start or is not running.")
            print("Attempting to reset the mon0 interface...")

            # 1. Delete the (potentially broken) interface
            self.run_command("iw dev mon0 del")
            # 2. Re-add the monitor interface
            self.run_command("iw phy phy0 interface add mon0 type monitor")
            # 3. Bring the interface up
            _, setup_error = self.run_command("ip link set mon0 up")

            if setup_error:
                print(f"❌ Failed to bring mon0 up. Error: {setup_error}")
                raise RuntimeError("Unable to bring mon0 up after tcpdump failure.")

            print("✅ Interface reset complete. Retrying tcpdump...")
            
            # Retry running the command
            self.run_command(command)
            time.sleep(1) # Give it another moment

            # Check one last time
            output, _ = self.run_command("pgrep -f 'tcpdump -i mon0'")
            
            if output:
                print(f"✅ tcpdump started successfully with PID {output} after interface reset.")
                tcpdump_started = True
            else:
                print("❌ tcpdump failed on the second attempt. Please check the router's logs.")
                raise RuntimeError("tcpdump failed to start after interface reset.")
        else:
            print(f"✅ tcpdump started successfully on the first attempt with PID {output}.")
            tcpdump_started = True
        
        if tcpdump_started and start_collector:
            self.start_pcap_collection()

        return tcpdump_started

    def kill_tcpdump(self):
        self.run_command("killall tcpdump")

    # Touched after a clock correction to re-seed sysfixtime's restore anchor.
    # Any regular file under /etc works (see sync_clock); /etc/banner is inert,
    # so bumping its mtime has no side effect beyond the one we want.
    CLOCK_ANCHOR = "/etc/banner"

    def sync_clock(self, tolerance=60):
        """Push this machine's time to the router, persistently.

        The router has no RTC and no WAN uplink, so ntpd can never sync it —
        system.ntp.enabled is set, but sysntpd isn't even running, and it would
        have nothing to reach if it were. On every boot /etc/init.d/sysfixtime
        restores the clock from the newest mtime under /etc.

        That is why a bare 'date -s' never sticks: it moves the clock but not
        the anchor, so the next boot drops straight back to whatever stale date
        the newest /etc file carries — the same "particular time", every time.
        Worse, it is self-perpetuating: any config written while the clock is
        wrong re-stamps the anchor with the wrong date. So we touch a file in
        /etc *after* setting the time, which makes the corrected time the new
        floor for subsequent boots.

        Only corrects when the drift exceeds `tolerance` seconds: /etc lives on
        flash, and there's no reason to burn a write on every session.

        Returns the remaining offset in seconds, or None if the sync failed.
        """
        try:
            offset = self.get_clock_offset()

            if abs(offset) <= tolerance:
                print(f"[Clock] Router is within {tolerance}s of this machine — leaving it alone.")
                return offset

            print(
                f"[Clock] Router clock is off by {offset:+.0f} s "
                f"({offset / 86400:+.1f} days). Correcting..."
            )
            epoch = round(time.time())
            _, error = self.run_command(
                f"date -s @{epoch} > /dev/null && touch {self.CLOCK_ANCHOR}"
            )
            if error:
                print(f"[Clock] ⚠️ Could not set the router clock: {error}")
                return offset

            remaining = self.get_clock_offset()
            print(f"[Clock] ✅ Router clock synced (residual {remaining:+.1f} s).")
            return remaining

        except Exception as e:
            # Never fatal: a wrong clock is handled downstream by shifting the
            # session window with get_clock_offset() instead.
            print(f"[Clock] ⚠️ Clock sync failed: {e}")
            return None

    def get_clock_offset(self):
        """Seconds to add to a host epoch to express it in the router's clock.

        Packet timestamps in the captured pcaps come from the router, which on
        an OpenWrt box with no RTC and no NTP sync can sit months away from the
        host clock (observed: 331 days behind). Anything that compares capture
        timestamps against a host-side time.time() has to shift by this first,
        or it will match nothing at all.

        Returns 0.0 if the offset can't be determined, which leaves callers
        with the old same-clock assumption rather than a wrong correction.
        """
        if not self.client:
            return 0.0

        before = time.time()
        output, _ = self.run_command('date +%s')
        after = time.time()

        if not output:
            return 0.0

        try:
            remote_epoch = float(output.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(f"[Clock] Could not parse remote date output: {output!r}")
            return 0.0

        # 'date +%s' floors to the second, so add half a second back; the host
        # side is the midpoint of the SSH round trip.
        offset = (remote_epoch + 0.5) - (before + after) / 2
        print(f"[Clock] Router clock offset vs host: {offset:+.1f} s")
        return offset

    def start_pcap_collection(self):
        """Starts the background thread to monitor and download pcap files."""
        if self.pcap_collector_thread and self.pcap_collector_thread.is_alive():
            print("⚠️ Collector thread is already running.")
            return

        print("🚀 Starting background pcap collector...")
        os.makedirs(self.local_pcap_dir, exist_ok=True) # Ensure local directory exists
        self._stop_event.clear()
        self._processed_files.clear()

        if not self._is_fast:
            self.pcap_collector_thread = threading.Thread(
                target=self._pcap_collection_loop,
                daemon=True # Allows main program to exit even if thread is running
            )
        else:
            self.pcap_collector_thread = threading.Thread(
                target=self._pcap_direct_download_loop,
                daemon=True # Allows main program to exit even if thread is running
            )
        self.pcap_collector_thread.start()

    def stop_pcap_collection(self):
        """
        Stops the background thread, kills the remote tcpdump process,
        and performs a final sweep to collect any remaining pcap files.
        This method is idempotent and safe to call multiple times.
        """
        if self._is_stopping:
            print("Shutdown already in progress.")
            return # Prevent duplicate runs
        
        self._is_stopping = True
        print("🛑 Stopping background pcap collector...")

        try:
            # 1. Stop the background thread first
            if self.pcap_collector_thread and self.pcap_collector_thread.is_alive():
                self._stop_event.set()
                self.pcap_collector_thread.join(timeout=5)
            
            # 2. Stop the remote tcpdump process. This finalizes the last file.
            self.kill_tcpdump()
            time.sleep(1) # Give a moment for the process to die

            # 3. Perform the "final sweep" for any remaining files
            print("[Final Sweep] Checking for leftover pcap files...")
            try:
                sftp = self.client.open_sftp()
                all_remote_files = sftp.listdir('/tmp/')
                pcap_files_to_collect = [f for f in all_remote_files if f.startswith('bfm_capture')]

                if not pcap_files_to_collect:
                    print("[Final Sweep] No remaining files found.")
                
                for filename in pcap_files_to_collect:
                    remote_path = f"/tmp/{filename}"
                    if remote_path not in self._processed_files:
                        local_path = os.path.join(self.local_pcap_dir, self.get_filename())
                        
                        print(f"[Final Sweep] Collecting remaining file: {filename}...")
                        
                        sftp.get(remote_path, local_path)
                        sftp.remove(remote_path)
                        self._processed_files.add(remote_path)
                        self._collected_files.add(local_path)
                        
                        print(f"[Final Sweep] ✅ Collected {local_path}.")
                sftp.close()
            except Exception as e:
                print(f"[Final Sweep] ❌ Error during final collection: {e}")

            print("Collector and remote tcpdump stopped.")

        finally:
            # Reset the flag so the method can be called again if needed in a new session
            self._is_stopping = False

    def _pcap_collection_loop(self):
        """
        The core logic for the background thread. This is where the magic happens.
        It finds the completed (older) pcap file, downloads it, and deletes it.
        """
        sftp = None
        try:
            sftp = self.client.open_sftp()
            print("[Collector Thread] Started successfully.")
        except Exception as e:
            print(f"[Collector Thread] Failed to open SFTP session: {e}")
            return # Exit thread if SFTP fails

        while not self._stop_event.is_set():
            try:
                remote_files = sftp.listdir('/tmp/')
                pcap_files = [f for f in remote_files if f.startswith('bfm_capture')]
                
                # We need two files to determine which one is "complete"
                if not pcap_files:
                    time.sleep(2)
                    continue

                file_stats = []
                for f in pcap_files:
                    path = f"/tmp/{f}"
                    stat = sftp.stat(path)
                    file_stats.append({'path': path, 'mtime': stat.st_mtime, 'size': stat.st_size})

                # Sort by modification time to find the oldest file
                file_stats.sort(key=lambda x: x['mtime'])

                # Only files tcpdump has rotated away from are safe to take:
                # everything but the newest. The newest is the one tcpdump still
                # holds open, and grabbing it mid-capture gives a truncated file
                # *and* — because we remove it afterwards — leaves tcpdump
                # writing the rest of the session into an unlinked inode, so
                # that data is lost outright. Judging "finished" by an unchanged
                # size/mtime is not safe either: tcpdump buffers its writes, so a
                # low BFM report rate makes an actively-written file look idle.
                # The still-open file is picked up by stop_pcap_collection()'s
                # final sweep once tcpdump has exited and flushed it.
                candidates = file_stats[:-1] if len(file_stats) >= 2 else []

                for completed_file in candidates:
                    if completed_file['path'] in self._processed_files:
                        continue

                    local_path = os.path.join(self.local_pcap_dir, self.get_filename())
                    print(f"[Collector Thread] New file found: {completed_file['path']}. Downloading...")
                    
                    # 1. Download the file
                    sftp.get(completed_file['path'], local_path)
                    
                    # 2. Delete the remote file after successful download
                    sftp.remove(completed_file['path'])
                    
                    # 3. Mark it as processed
                    self._processed_files.add(completed_file['path'])
                    self._collected_files.add(local_path)
                    print(f"[Collector Thread] ✅ Download complete. Deleted remote file.")

            except Exception as e:
                print(f"[Collector Thread] An error occurred: {e}")
                # If the connection drops, the thread will eventually exit.
            
            time.sleep(3) # Check for new files every 3 seconds

        sftp.close()
        print("[Collector Thread] Exiting.")

    def _pcap_direct_download_loop(self):
        """
        Continuously downloads a snapshot of a single active file every second.

        WARNING: This approach is highly inefficient and leads to massive data
        duplication and unmanaged remote file growth. It is provided for
        educational purposes to demonstrate the concept, but the robust
        rotation method is strongly recommended for production use.
        """
        sftp = None
        REMOTE_FILENAME = 'bfm_capture' # The single, non-rotating filename
        REMOTE_PATH = f'/tmp/{REMOTE_FILENAME}'

        try:
            sftp = self.client.open_sftp()
            print("[Direct Download Thread] Started successfully.")
        except Exception as e:
            print(f"[Direct Download Thread] Failed to open SFTP session: {e}")
            return

        while not self._stop_event.is_set():
            try:
                # Generate a unique local filename for each snapshot
                timestamp = int(time.time())
                local_filename = self.get_filename()
                local_path = os.path.join(self.local_pcap_dir, local_filename)

                # 1. Attempt to download the current state of the file
                print(f"[Direct Download Thread] Attempting to download snapshot of {REMOTE_PATH}...")
                sftp.get(REMOTE_PATH, local_path)
                
                # 2. Add to collected files list. DO NOT DELETE THE REMOTE FILE.
                self._collected_files.add(local_path)
                print(f"[Direct Download Thread] ✅ Snapshot downloaded to {local_path}")

            except FileNotFoundError:
                # This is expected if the loop runs before tcpdump creates the file
                print(f"[Direct Download Thread] Remote file {REMOTE_PATH} not found yet. Waiting...")
            except Exception as e:
                print(f"[Direct Download Thread] An error occurred: {e}")
                # Break the loop on significant errors like a connection drop
                break
            
            time.sleep(1) # Wait for one second before the next download

        if sftp:
            sftp.close()
        print("[Direct Download Thread] Exiting.")

    def get_collected_files(self):
        return self._collected_files

    def close(self):
        """Stops background processes and closes the SSH connection."""
        self.stop_pcap_collection()
        if self.client:
            self.client.close()
            self.client = None
            print("🔌 Connection to router closed.")




def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--host', '-H',
        default = '192.168.1.1'
    )

    parser.add_argument(
        '--username', '-u',
        default = 'root'
    )

    parser.add_argument(
        '--password', '-p',
        default = '123456'
    )

    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    collector = BFMCollector(args.host, args.username, args.password)
    collector.connect()
    collector.run_iperf3()
    collector.fast()
    time.sleep(5)
    collector.run_tcpdump()
    time.sleep(20)
    collector.kill_tcpdump()
    collector.kill_iperf3()

    collector.close()
