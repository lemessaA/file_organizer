import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class JdbcExample {
    public static void main(String[] args) {
        String url = "jdbc:postgresql://localhost:5432/jdbc_db";
        String user = "postgres";
        String password = "qaws";

        // Try-with-resources ensures automatic closing
        try (Connection conn = DriverManager.getConnection(url, user, password)) {
            System.out.println("Connected successfully");

            String query = "SELECT id, name FROM users";
            try (PreparedStatement stmt = conn.prepareStatement(query);
                 ResultSet rs = stmt.executeQuery()) {

                while (rs.next()) {
                    int id = rs.getInt("id");
                    String name = rs.getString("name");
                    System.out.println(id + " - " + name);
                }
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
