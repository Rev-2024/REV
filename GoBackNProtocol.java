import java.util.Arrays;
import java.util.Random;
class GoBackNProtocol {
 private final int WINDOW_SIZE; 
 private final int TOTAL_FRAMES; 
 private int nextFrameToSend = 0; 
 private int frameExpectedByReceiver = 0; 
 public GoBackNProtocol(int totalFrames, int windowSize) {
 this.TOTAL_FRAMES = totalFrames;
 this.WINDOW_SIZE = windowSize;
 }
 public void sendFrames() {
 Random random = new Random(); 
 while (nextFrameToSend < TOTAL_FRAMES) {
 for (int i = 0; i < WINDOW_SIZE && nextFrameToSend < TOTAL_FRAMES; i++) {
 System.out.println("Sender: Sending frame " + nextFrameToSend);
 nextFrameToSend++; 
 }
 if (random.nextBoolean()) { 
 frameExpectedByReceiver += WINDOW_SIZE; 
 System.out.println("Receiver: ACK received for frames up to " + frameExpectedByReceiver);
} else { 
    System.out.println("Receiver: ACK lost, resending frames from " +
   frameExpectedByReceiver);
    nextFrameToSend = frameExpectedByReceiver; 
    }
    }

    System.out.println("All frames sent and acknowledged successfully.");
    }
    
    public static void main(String[] args) {
    int totalFrames = 10; 
    int windowSize = 4; 
    GoBackNProtocol protocol = new GoBackNProtocol(totalFrames, windowSize);
    protocol.sendFrames();
    }
   } 